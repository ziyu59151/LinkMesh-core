#include <PCU.h>
#include <agm.h>
#include <apf.h>
#include <apfMDS.h>
#include <apfMesh2.h>
#include <gmi.h>
#include <gmi_analytic.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr std::array<char, 8> kMagic{
    'L', 'M', '4', 'X', 'C', 'H', 'G', '\0'
};
constexpr std::uint32_t kSchemaVersionV1 = 1;
constexpr std::uint32_t kSchemaVersionV2 = 2;
constexpr std::uint32_t kEndianMarker = 0x01020304;
constexpr std::int32_t kLinearCurve = 1;

struct Counts {
  std::uint64_t nodes = 0;
  std::uint64_t triangles = 0;
  std::uint64_t mesh_edges = 0;
  std::uint64_t geometric_vertices = 0;
  std::uint64_t geometric_edges = 0;
  std::uint64_t geometric_loops = 0;
  std::uint64_t geometric_faces = 0;
};

struct Classification {
  std::int32_t dimension = -1;
  std::int32_t tag = -1;
};

struct Node {
  double r = 0.0;
  double z = 0.0;
};

struct Triangle {
  std::array<std::uint64_t, 3> nodes{};
};

struct GeometricVertex {
  std::int32_t tag = 0;
  double r = 0.0;
  double z = 0.0;
  double y = 0.0;
};

struct GeometricEdge {
  std::int32_t tag = 0;
  std::int32_t begin_vertex_tag = 0;
  std::int32_t end_vertex_tag = 0;
  std::int32_t curve_type = 0;
};

struct GeometricLoop {
  std::int32_t tag = 0;
  std::vector<std::int32_t> edge_tags;
};

struct GeometricFace {
  std::int32_t tag = 0;
  std::vector<std::int32_t> loop_tags;
};

struct MeshEdge {
  std::uint64_t node0 = 0;
  std::uint64_t node1 = 0;
  Classification classification;
};

struct Exchange {
  std::uint32_t schema_version = 0;
  Counts counts;
  std::array<std::int32_t, 4> special_loop_tags{};
  std::filesystem::path output_base;
  std::vector<Node> nodes;
  std::vector<Triangle> triangles;
  std::vector<GeometricVertex> geometric_vertices;
  std::vector<GeometricEdge> geometric_edges;
  std::vector<GeometricLoop> geometric_loops;
  std::vector<GeometricFace> geometric_faces;
  std::vector<Classification> vertex_classifications;
  std::vector<MeshEdge> mesh_edges;
  std::vector<Classification> triangle_classifications;
};

struct PointData {
  std::array<double, 3> coordinates{};
};

struct LineData {
  std::array<double, 3> begin{};
  std::array<double, 3> end{};
};

class StageError : public std::runtime_error {
 public:
  StageError(std::string stage, std::string message)
      : std::runtime_error(std::move(message)),
        stage_(std::move(stage)) {}

  const std::string& stage() const noexcept {
    return stage_;
  }

 private:
  std::string stage_;
};

std::string percent_encode(const std::string& value) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string encoded;
  encoded.reserve(value.size());

  for (unsigned char character : value) {
    const bool safe =
        (character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9') ||
        character == '-' || character == '_' || character == '.' ||
        character == '/' || character == ':' || character == '\\';

    if (safe) {
      encoded.push_back(static_cast<char>(character));
    } else {
      encoded.push_back('%');
      encoded.push_back(kHex[(character >> 4U) & 0x0FU]);
      encoded.push_back(kHex[character & 0x0FU]);
    }
  }
  return encoded;
}

std::string single_line(std::string value) {
  std::replace(value.begin(), value.end(), '\n', ' ');
  std::replace(value.begin(), value.end(), '\r', ' ');
  return value;
}

template <typename T>
T read_scalar(std::istream& stream, const char* field_name) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream) {
    throw StageError(
        "exchange_read",
        std::string("truncated field: ") + field_name);
  }
  return value;
}

template <typename T>
std::vector<T> checked_vector(std::uint64_t count, const char* field_name) {
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    throw StageError(
        "exchange_validate",
        std::string(field_name) + " count exceeds size_t");
  }
  return std::vector<T>(static_cast<std::size_t>(count));
}

void require(bool condition, const char* stage, const std::string& message) {
  if (!condition) {
    throw StageError(stage, message);
  }
}

void read_exact(
    std::istream& stream,
    char* destination,
    std::size_t byte_count,
    const char* field_name) {
  if (byte_count == 0) {
    return;
  }
  stream.read(destination, static_cast<std::streamsize>(byte_count));
  if (!stream) {
    throw StageError(
        "exchange_read",
        std::string("truncated field: ") + field_name);
  }
}

Exchange read_exchange(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw StageError(
        "exchange_open",
        "cannot open exchange file: " + path.string());
  }

  std::array<char, 8> magic{};
  read_exact(stream, magic.data(), magic.size(), "magic");
  require(
      magic == kMagic,
      "exchange_validate",
      "invalid LinkMesh exchange magic");

  const auto schema_version =
      read_scalar<std::uint32_t>(stream, "schema_version");
  const auto endian_marker =
      read_scalar<std::uint32_t>(stream, "endian_marker");

  require(
      schema_version == kSchemaVersionV1 ||
          schema_version == kSchemaVersionV2,
      "exchange_validate",
      "unsupported schema_version=" + std::to_string(schema_version));
  require(
      endian_marker == kEndianMarker,
      "exchange_validate",
      "invalid endian marker or non-little-endian exchange");

  Exchange exchange;
  exchange.schema_version = schema_version;
  exchange.counts.nodes = read_scalar<std::uint64_t>(stream, "N");
  exchange.counts.triangles = read_scalar<std::uint64_t>(stream, "T");
  exchange.counts.mesh_edges = read_scalar<std::uint64_t>(stream, "E");
  exchange.counts.geometric_vertices =
      read_scalar<std::uint64_t>(stream, "GV");
  exchange.counts.geometric_edges =
      read_scalar<std::uint64_t>(stream, "GE");
  exchange.counts.geometric_loops =
      read_scalar<std::uint64_t>(stream, "GL");
  exchange.counts.geometric_faces =
      read_scalar<std::uint64_t>(stream, "GF");

  for (auto& tag : exchange.special_loop_tags) {
    tag = read_scalar<std::int32_t>(stream, "special_loop_tag");
  }

  const auto output_length =
      read_scalar<std::uint64_t>(stream, "output_path_length");
  require(
      output_length > 0,
      "exchange_validate",
      "output SMB base path is empty");
  require(
      output_length <= 1024U * 1024U,
      "exchange_validate",
      "output SMB base path is unreasonably long");

  std::string output_bytes(static_cast<std::size_t>(output_length), '\0');
  read_exact(
      stream,
      output_bytes.data(),
      output_bytes.size(),
      "output_base");
  require(
      output_bytes.find('\0') == std::string::npos,
      "exchange_validate",
      "output SMB base path contains NUL");
  exchange.output_base = std::filesystem::path(output_bytes);

  exchange.nodes =
      checked_vector<Node>(exchange.counts.nodes, "nodes");
  for (auto& node : exchange.nodes) {
    node.r = read_scalar<double>(stream, "node.r");
    node.z = read_scalar<double>(stream, "node.z");
  }

  exchange.triangles =
      checked_vector<Triangle>(exchange.counts.triangles, "triangles");
  for (auto& triangle : exchange.triangles) {
    for (auto& node_index : triangle.nodes) {
      node_index = read_scalar<std::uint64_t>(
          stream, "triangle.node_index");
    }
  }

  exchange.geometric_vertices = checked_vector<GeometricVertex>(
      exchange.counts.geometric_vertices,
      "geometric_vertices");
  for (auto& vertex : exchange.geometric_vertices) {
    vertex.tag =
        read_scalar<std::int32_t>(stream, "geometric_vertex.tag");
    vertex.r = read_scalar<double>(stream, "geometric_vertex.r");
    vertex.z = read_scalar<double>(stream, "geometric_vertex.z");
    vertex.y = read_scalar<double>(stream, "geometric_vertex.y");
  }

  exchange.geometric_edges = checked_vector<GeometricEdge>(
      exchange.counts.geometric_edges,
      "geometric_edges");
  for (auto& edge : exchange.geometric_edges) {
    edge.tag = read_scalar<std::int32_t>(
        stream, "geometric_edge.tag");
    edge.begin_vertex_tag = read_scalar<std::int32_t>(
        stream, "geometric_edge.begin_vertex_tag");
    edge.end_vertex_tag = read_scalar<std::int32_t>(
        stream, "geometric_edge.end_vertex_tag");
    edge.curve_type = read_scalar<std::int32_t>(
        stream, "geometric_edge.curve_type");
  }

  exchange.geometric_loops = checked_vector<GeometricLoop>(
      exchange.counts.geometric_loops,
      "geometric_loops");
  for (auto& loop : exchange.geometric_loops) {
    loop.tag =
        read_scalar<std::int32_t>(stream, "geometric_loop.tag");
    const auto edge_count = read_scalar<std::uint64_t>(
        stream, "geometric_loop.edge_count");
    loop.edge_tags = checked_vector<std::int32_t>(
        edge_count, "geometric_loop.edge_tags");
    for (auto& edge_tag : loop.edge_tags) {
      edge_tag = read_scalar<std::int32_t>(
          stream, "geometric_loop.edge_tag");
    }
  }

  exchange.geometric_faces = checked_vector<GeometricFace>(
      exchange.counts.geometric_faces,
      "geometric_faces");
  for (auto& face : exchange.geometric_faces) {
    face.tag =
        read_scalar<std::int32_t>(stream, "geometric_face.tag");
    const auto loop_count = read_scalar<std::uint64_t>(
        stream, "geometric_face.loop_count");
    face.loop_tags = checked_vector<std::int32_t>(
        loop_count, "geometric_face.loop_tags");
    for (auto& loop_tag : face.loop_tags) {
      loop_tag = read_scalar<std::int32_t>(
          stream, "geometric_face.loop_tag");
    }
  }

  exchange.vertex_classifications = checked_vector<Classification>(
      exchange.counts.nodes,
      "vertex_classifications");
  for (auto& classification : exchange.vertex_classifications) {
    classification.dimension = read_scalar<std::int32_t>(
        stream, "vertex_classification.dimension");
    classification.tag = read_scalar<std::int32_t>(
        stream, "vertex_classification.tag");
  }

  exchange.mesh_edges = checked_vector<MeshEdge>(
      exchange.counts.mesh_edges,
      "mesh_edges");
  for (auto& edge : exchange.mesh_edges) {
    edge.node0 =
        read_scalar<std::uint64_t>(stream, "mesh_edge.node0");
    edge.node1 =
        read_scalar<std::uint64_t>(stream, "mesh_edge.node1");
    edge.classification.dimension = read_scalar<std::int32_t>(
        stream, "mesh_edge.classification.dimension");
    edge.classification.tag = read_scalar<std::int32_t>(
        stream, "mesh_edge.classification.tag");
  }

  exchange.triangle_classifications = checked_vector<Classification>(
      exchange.counts.triangles,
      "triangle_classifications");
  for (auto& classification : exchange.triangle_classifications) {
    classification.dimension = read_scalar<std::int32_t>(
        stream, "triangle_classification.dimension");
    classification.tag = read_scalar<std::int32_t>(
        stream, "triangle_classification.tag");
  }

  char trailing = '\0';
  stream.read(&trailing, 1);
  require(
      stream.eof(),
      "exchange_validate",
      "exchange file contains trailing bytes");

  return exchange;
}

template <typename Entity>
std::map<std::int32_t, const Entity*> unique_positive_tags(
    const std::vector<Entity>& entities,
    const char* entity_name) {
  std::map<std::int32_t, const Entity*> result;
  for (const auto& entity : entities) {
    require(
        entity.tag > 0,
        "exchange_validate",
        std::string(entity_name) + " tag must be positive");
    const auto inserted = result.emplace(entity.tag, &entity).second;
    require(
        inserted,
        "exchange_validate",
        std::string("duplicate ") + entity_name +
            " tag=" + std::to_string(entity.tag));
  }
  return result;
}

std::pair<std::uint64_t, std::uint64_t> edge_key(
    std::uint64_t first,
    std::uint64_t second) {
  return first < second
      ? std::make_pair(first, second)
      : std::make_pair(second, first);
}

double signed_double_area(
    const Node& p0,
    const Node& p1,
    const Node& p2) {
  return
      (p1.r - p0.r) * (p2.z - p0.z) -
      (p1.z - p0.z) * (p2.r - p0.r);
}

void validate_classification(
    const Classification& classification,
    const std::set<std::int32_t>& vertex_tags,
    const std::set<std::int32_t>& edge_tags,
    const std::set<std::int32_t>& face_tags,
    const char* entity_name,
    bool mesh_vertex) {
  bool valid = false;
  if (mesh_vertex && classification.dimension == 0) {
    valid = vertex_tags.count(classification.tag) == 1;
  } else if (
      classification.dimension == 1 &&
      (mesh_vertex || std::strcmp(entity_name, "mesh edge") == 0)) {
    valid = edge_tags.count(classification.tag) == 1;
  } else if (classification.dimension == 2) {
    valid = face_tags.count(classification.tag) == 1;
  }

  require(
      valid,
      "exchange_validate",
      std::string("invalid ") + entity_name +
          " classification dim=" +
          std::to_string(classification.dimension) +
          " tag=" + std::to_string(classification.tag));
}

void validate_exchange(const Exchange& exchange) {
  const auto& counts = exchange.counts;
  require(counts.nodes >= 3, "exchange_validate", "N must be >= 3");
  require(counts.triangles > 0, "exchange_validate", "T must be > 0");
  require(
      counts.geometric_vertices >= 3,
      "exchange_validate",
      "GV must be >= 3");
  require(
      counts.geometric_edges >= 3,
      "exchange_validate",
      "GE must be >= 3");
  require(
      counts.geometric_loops >= 1,
      "exchange_validate",
      "GL must be >= 1");
  require(
      counts.geometric_faces >= 1,
      "exchange_validate",
      "GF must be >= 1");
  if (exchange.schema_version == kSchemaVersionV1) {
    require(
        counts.geometric_edges == counts.geometric_vertices,
        "exchange_validate",
        "schema v1 single-loop GE must equal GV");
    require(
        counts.geometric_loops == 1,
        "exchange_validate",
        "schema v1 requires GL=1");
    require(
        counts.geometric_faces == 1,
        "exchange_validate",
        "schema v1 requires GF=1");
  }

  require(
      exchange.output_base.extension() == ".smb",
      "exchange_validate",
      "output base must end with .smb");
  require(
      !exchange.output_base.stem().string().empty(),
      "exchange_validate",
      "output base filename is empty");
  require(
      exchange.output_base.stem().string().back() != '0',
      "exchange_validate",
      "output path appears to be an already ranked rank-0 path");

  for (const auto& node : exchange.nodes) {
    require(
        std::isfinite(node.r) && std::isfinite(node.z),
        "exchange_validate",
        "node contains NaN or Inf");
  }

  const auto vertex_by_tag = unique_positive_tags(
      exchange.geometric_vertices, "geometric vertex");
  const auto edge_by_tag = unique_positive_tags(
      exchange.geometric_edges, "geometric edge");
  const auto loop_by_tag = unique_positive_tags(
      exchange.geometric_loops, "geometric loop");
  const auto face_by_tag = unique_positive_tags(
      exchange.geometric_faces, "geometric face");

  std::set<std::int32_t> vertex_tags;
  std::set<std::int32_t> edge_tags;
  std::set<std::int32_t> face_tags;
  for (const auto& item : vertex_by_tag) {
    vertex_tags.insert(item.first);
    require(
        std::isfinite(item.second->r) &&
            std::isfinite(item.second->z) &&
            std::isfinite(item.second->y),
        "exchange_validate",
        "geometric vertex contains NaN or Inf");
    require(
        item.second->y == 0.0,
        "exchange_validate",
        "geometric vertex y must be zero");
  }
  for (const auto& item : edge_by_tag) {
    edge_tags.insert(item.first);
    const auto* edge = item.second;
    require(
        edge->curve_type == kLinearCurve,
        "exchange_validate",
        "LinkMesh exchange only accepts LIN geometric edges");
    require(
        edge->begin_vertex_tag != edge->end_vertex_tag,
        "exchange_validate",
        "geometric edge has identical endpoint tags");
    require(
        vertex_by_tag.count(edge->begin_vertex_tag) == 1 &&
            vertex_by_tag.count(edge->end_vertex_tag) == 1,
        "exchange_validate",
        "geometric edge has dangling vertex reference");
  }
  for (const auto& item : face_by_tag) {
    face_tags.insert(item.first);
  }

  std::map<std::int32_t, int> edge_loop_uses;
  std::map<std::int32_t, int> loop_face_uses;
  for (const auto& loop : exchange.geometric_loops) {
    require(
        loop.edge_tags.size() >= 3,
        "exchange_validate",
        "geometric loop must contain at least three edges");
    require(
        std::set<std::int32_t>(
            loop.edge_tags.begin(), loop.edge_tags.end()).size() ==
            loop.edge_tags.size(),
        "exchange_validate",
        "geometric loop contains duplicate edge tags");
    for (std::size_t index = 0; index < loop.edge_tags.size(); ++index) {
      require(
          edge_by_tag.count(loop.edge_tags[index]) == 1,
          "exchange_validate",
          "geometric loop has dangling edge reference");
      ++edge_loop_uses[loop.edge_tags[index]];
      const auto* current = edge_by_tag.at(loop.edge_tags[index]);
      const auto* next = edge_by_tag.at(
          loop.edge_tags[(index + 1) % loop.edge_tags.size()]);
      require(
          current->end_vertex_tag == next->begin_vertex_tag,
          "exchange_validate",
          "geometric loop is not directed and continuous");
    }
  }
  for (const auto edge_tag : edge_tags) {
    require(
        edge_loop_uses[edge_tag] == 1,
        "exchange_validate",
        "each geometric edge must belong to exactly one loop");
  }

  for (const auto& face : exchange.geometric_faces) {
    require(
        !face.loop_tags.empty(),
        "exchange_validate",
        "geometric face must reference at least one loop");
    require(
        std::set<std::int32_t>(
            face.loop_tags.begin(), face.loop_tags.end()).size() ==
            face.loop_tags.size(),
        "exchange_validate",
        "geometric face contains duplicate loop tags");
    for (const auto loop_tag : face.loop_tags) {
      require(
          loop_by_tag.count(loop_tag) == 1,
          "exchange_validate",
          "geometric face has dangling loop reference");
      ++loop_face_uses[loop_tag];
    }
  }
  for (const auto& item : loop_by_tag) {
    require(
        loop_face_uses[item.first] == 1 ||
            loop_face_uses[item.first] == 2,
        "exchange_validate",
        "each geometric loop must be referenced by one or two faces");
  }

  if (exchange.schema_version == kSchemaVersionV1) {
    const auto& loop = exchange.geometric_loops.front();
    const auto& face = exchange.geometric_faces.front();
    require(
        loop.edge_tags.size() == exchange.geometric_edges.size() &&
            std::set<std::int32_t>(
                loop.edge_tags.begin(), loop.edge_tags.end()) == edge_tags,
        "exchange_validate",
        "schema v1 unique loop must reference every edge once");
    require(
        face.loop_tags.size() == 1 &&
            face.loop_tags.front() == loop.tag,
        "exchange_validate",
        "schema v1 unique face must reference unique loop");
    require(
        exchange.special_loop_tags ==
            std::array<std::int32_t, 4>{-1, loop.tag, -1, -1},
        "exchange_validate",
        "schema v1 special loop tags are invalid");
  } else {
    require(
        exchange.special_loop_tags ==
            std::array<std::int32_t, 4>{-1, -1, -1, -1},
        "exchange_validate",
        "schema v2 special loop tags must all be -1");
  }

  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t>
      mesh_edge_indices;
  std::pair<std::uint64_t, std::uint64_t> previous_edge{0, 0};
  bool have_previous = false;

  for (std::size_t index = 0;
       index < exchange.mesh_edges.size();
       ++index) {
    const auto& edge = exchange.mesh_edges[index];
    require(
        edge.node0 < edge.node1,
        "exchange_validate",
        "mesh edge endpoints must satisfy node0 < node1");
    require(
        edge.node1 < counts.nodes,
        "exchange_validate",
        "mesh edge node index is out of range");

    const auto key = std::make_pair(edge.node0, edge.node1);
    if (have_previous) {
      require(
          previous_edge < key,
          "exchange_validate",
          "mesh edge records must be unique and lexicographically sorted");
    }
    previous_edge = key;
    have_previous = true;
    mesh_edge_indices.emplace(key, index);

    validate_classification(
        edge.classification,
        vertex_tags,
        edge_tags,
        face_tags,
        "mesh edge",
        false);
    require(
        edge.classification.dimension == 1 ||
            edge.classification.dimension == 2,
        "exchange_validate",
        "mesh edge classification dimension must be 1 or 2");
  }

  std::map<std::pair<std::uint64_t, std::uint64_t>, int>
      triangle_edge_counts;
  for (std::size_t index = 0;
       index < exchange.triangles.size();
       ++index) {
    const auto& triangle = exchange.triangles[index];
    const auto n0 = triangle.nodes[0];
    const auto n1 = triangle.nodes[1];
    const auto n2 = triangle.nodes[2];

    require(
        n0 < counts.nodes && n1 < counts.nodes && n2 < counts.nodes,
        "exchange_validate",
        "triangle node index is out of range");
    require(
        n0 != n1 && n1 != n2 && n0 != n2,
        "exchange_validate",
        "triangle contains duplicate node indices");

    const double area = signed_double_area(
        exchange.nodes[static_cast<std::size_t>(n0)],
        exchange.nodes[static_cast<std::size_t>(n1)],
        exchange.nodes[static_cast<std::size_t>(n2)]);
    require(
        std::isfinite(area) && std::abs(area) > 2.0e-12,
        "exchange_validate",
        "triangle is degenerate or has area <= 1e-12");

    for (const auto& key : {
             edge_key(n0, n1),
             edge_key(n1, n2),
             edge_key(n2, n0)}) {
      require(
          mesh_edge_indices.count(key) == 1,
          "exchange_validate",
          "triangle references a mesh edge absent from the edge table");
      ++triangle_edge_counts[key];
    }

    const auto& classification =
        exchange.triangle_classifications[index];
    validate_classification(
        classification,
        vertex_tags,
        edge_tags,
        face_tags,
        "triangle",
        false);
    require(
        classification.dimension == 2,
        "exchange_validate",
        "every triangle must classify on a geometric face");
  }

  require(
      triangle_edge_counts.size() == exchange.mesh_edges.size(),
      "exchange_validate",
      "mesh edge table contains edges unused by triangles");
  for (const auto& item : triangle_edge_counts) {
    require(
        item.second == 1 || item.second == 2,
        "exchange_validate",
        "mesh contains a non-manifold edge");
  }

  for (const auto& classification :
       exchange.vertex_classifications) {
    validate_classification(
        classification,
        vertex_tags,
        edge_tags,
        face_tags,
        "mesh vertex",
        true);
    if (exchange.schema_version == kSchemaVersionV2) {
      require(
          classification.dimension == 0 ||
              classification.dimension == 2,
          "exchange_validate",
          "schema v2 mesh vertex classification dimension must be 0 or 2");
    }
  }
}

void vertex_function(
    const double[2],
    double xyz[3],
    void* data) {
  const auto* point = static_cast<const PointData*>(data);
  xyz[0] = point->coordinates[0];
  xyz[1] = point->coordinates[1];
  xyz[2] = point->coordinates[2];
}

void line_function(
    const double parameter[2],
    double xyz[3],
    void* data) {
  const auto* line = static_cast<const LineData*>(data);
  const double t = parameter[0];
  for (int dimension = 0; dimension < 3; ++dimension) {
    xyz[dimension] =
        line->begin[dimension] +
        t * (line->end[dimension] - line->begin[dimension]);
  }
}

void face_function(
    const double parameter[2],
    double xyz[3],
    void*) {
  xyz[0] = parameter[0];
  xyz[1] = parameter[1];
  xyz[2] = 0.0;
}

void reparam_zero(const double[2], double to[2], void*) {
  to[0] = 0.0;
  to[1] = 0.0;
}

void reparam_one(const double[2], double to[2], void*) {
  to[0] = 1.0;
  to[1] = 0.0;
}

void edge_to_face_reparam(
    const double from[2],
    double to[2],
    void* data) {
  const auto* line = static_cast<const LineData*>(data);
  const double t = from[0];
  to[0] =
      line->begin[0] +
      t * (line->end[0] - line->begin[0]);
  to[1] =
      line->begin[1] +
      t * (line->end[1] - line->begin[1]);
}

agm_bdry add_boundary(gmi_model* model, gmi_ent* entity) {
  return agm_add_bdry(
      gmi_analytic_topo(model),
      agm_from_gmi(entity));
}

agm_use add_use(
    gmi_model* model,
    agm_bdry boundary,
    int adjacent_tag) {
  agm* topology = gmi_analytic_topo(model);
  const int dimension =
      agm_dim_from_type(agm_bounds(topology, boundary).type);
  gmi_ent* adjacent =
      gmi_find(model, dimension - 1, adjacent_tag);
  require(
      adjacent != nullptr,
      "model_create",
      "cannot find adjacent geometric entity tag=" +
          std::to_string(adjacent_tag));
  return agm_add_use(
      topology,
      boundary,
      agm_from_gmi(adjacent));
}

void add_edge_topology(
    gmi_model* model,
    gmi_ent* edge,
    int begin_tag,
    int end_tag) {
  const agm_bdry boundary = add_boundary(model, edge);

  const agm_use begin_use = add_use(model, boundary, begin_tag);
  gmi_add_analytic_reparam(
      model, begin_use, reparam_zero, nullptr);

  const agm_use end_use = add_use(model, boundary, end_tag);
  gmi_add_analytic_reparam(
      model, end_use, reparam_one, nullptr);
}

void add_face_topology(
    gmi_model* model,
    gmi_ent* face,
    const std::vector<std::int32_t>& edge_tags,
    const std::map<std::int32_t, LineData*>& line_data_by_tag) {
  const agm_bdry boundary = add_boundary(model, face);
  for (const auto edge_tag : edge_tags) {
    const auto line_data_it = line_data_by_tag.find(edge_tag);
    require(
        line_data_it != line_data_by_tag.end() &&
            line_data_it->second != nullptr,
        "model_create",
        "cannot find callback data for geometric edge tag=" +
            std::to_string(edge_tag));

    const agm_use edge_use = add_use(model, boundary, edge_tag);
    gmi_add_analytic_reparam(
        model,
        edge_use,
        edge_to_face_reparam,
        line_data_it->second);
  }
}

struct NativeModel {
  gmi_model* model = nullptr;
  std::vector<PointData> point_data;
  std::vector<LineData> line_data;

  NativeModel() = default;
  NativeModel(const NativeModel&) = delete;
  NativeModel& operator=(const NativeModel&) = delete;
  NativeModel(NativeModel&&) = delete;
  NativeModel& operator=(NativeModel&&) = delete;

  ~NativeModel() {
    if (model != nullptr) {
      gmi_destroy(model);
    }
  }
};

void create_model(
    const Exchange& exchange,
    NativeModel& native) {
  native.model = gmi_make_analytic();
  require(
      native.model != nullptr,
      "model_create",
      "gmi_make_analytic returned null");

  // Resize once before registering callbacks. No subsequent operation changes
  // these vector capacities, so every callback data address remains stable
  // until NativeModel destroys the GMI model after APF/MDS mesh destruction.
  native.point_data.resize(exchange.geometric_vertices.size());
  native.line_data.resize(exchange.geometric_edges.size());

  std::map<std::int32_t, const GeometricVertex*> vertex_by_tag;
  for (const auto& vertex : exchange.geometric_vertices) {
    vertex_by_tag.emplace(vertex.tag, &vertex);
  }

  double r_min = std::numeric_limits<double>::infinity();
  double r_max = -std::numeric_limits<double>::infinity();
  double z_min = std::numeric_limits<double>::infinity();
  double z_max = -std::numeric_limits<double>::infinity();

  for (std::size_t index = 0;
       index < exchange.geometric_vertices.size();
       ++index) {
    const auto& source = exchange.geometric_vertices[index];
    auto& data = native.point_data[index];
    data.coordinates = {source.r, source.z, source.y};

    r_min = std::min(r_min, source.r);
    r_max = std::max(r_max, source.r);
    z_min = std::min(z_min, source.z);
    z_max = std::max(z_max, source.z);

    gmi_ent* entity = gmi_add_analytic(
        native.model,
        0,
        source.tag,
        vertex_function,
        nullptr,
        nullptr,
        &data);
    require(
        entity != nullptr,
        "model_create",
        "failed to create geometric vertex tag=" +
            std::to_string(source.tag));
  }

  require(
      std::isfinite(r_min) &&
          std::isfinite(r_max) &&
          std::isfinite(z_min) &&
          std::isfinite(z_max),
      "model_create",
      "geometric face parameter ranges contain non-finite values");
  require(
      r_max > r_min,
      "model_create",
      "geometric face R parameter range is degenerate");
  require(
      z_max > z_min,
      "model_create",
      "geometric face Z parameter range is degenerate");

  int edge_periodic = 0;
  double edge_range[2] = {0.0, 1.0};
  std::map<std::int32_t, LineData*> line_data_by_tag;

  for (std::size_t index = 0;
       index < exchange.geometric_edges.size();
       ++index) {
    const auto& source = exchange.geometric_edges[index];
    const auto* begin =
        vertex_by_tag.at(source.begin_vertex_tag);
    const auto* end =
        vertex_by_tag.at(source.end_vertex_tag);
    auto& data = native.line_data[index];
    data.begin = {begin->r, begin->z, begin->y};
    data.end = {end->r, end->z, end->y};

    const bool inserted =
        line_data_by_tag.emplace(source.tag, &data).second;
    require(
        inserted,
        "model_create",
        "duplicate callback data for geometric edge tag=" +
            std::to_string(source.tag));

    gmi_ent* entity = gmi_add_analytic(
        native.model,
        1,
        source.tag,
        line_function,
        &edge_periodic,
        &edge_range,
        &data);
    require(
        entity != nullptr,
        "model_create",
        "failed to create geometric edge tag=" +
            std::to_string(source.tag));

    add_edge_topology(
        native.model,
        entity,
        source.begin_vertex_tag,
        source.end_vertex_tag);
  }

  int face_periodic[2] = {0, 0};
  double face_ranges[2][2] = {
      {r_min, r_max},
      {z_min, z_max},
  };

  std::map<std::int32_t, const GeometricLoop*> loop_by_tag;
  for (const auto& loop : exchange.geometric_loops) {
    loop_by_tag.emplace(loop.tag, &loop);
  }

  for (const auto& face : exchange.geometric_faces) {
    gmi_ent* face_entity = gmi_add_analytic(
        native.model,
        2,
        face.tag,
        face_function,
        face_periodic,
        face_ranges,
        nullptr);
    require(
        face_entity != nullptr,
        "model_create",
        "failed to create geometric face tag=" +
            std::to_string(face.tag));

    std::vector<std::int32_t> face_edge_tags;
    for (const auto loop_tag : face.loop_tags) {
      const auto* loop = loop_by_tag.at(loop_tag);
      face_edge_tags.insert(
          face_edge_tags.end(),
          loop->edge_tags.begin(),
          loop->edge_tags.end());
    }
    add_face_topology(
        native.model,
        face_entity,
        face_edge_tags,
        line_data_by_tag);
  }
}

apf::ModelEntity* find_model_entity(
    apf::Mesh2* mesh,
    int dimension,
    int tag) {
  apf::ModelEntity* entity =
      mesh->findModelEntity(dimension, tag);
  require(
      entity != nullptr,
      "mesh_create",
      "cannot find model entity dim=" +
          std::to_string(dimension) +
          " tag=" + std::to_string(tag));
  return entity;
}

double edge_parameter(
    const Exchange& exchange,
    const GeometricEdge& edge,
    const Node& point) {
  std::map<std::int32_t, const GeometricVertex*> vertex_by_tag;
  for (const auto& vertex : exchange.geometric_vertices) {
    vertex_by_tag.emplace(vertex.tag, &vertex);
  }

  const auto* begin = vertex_by_tag.at(edge.begin_vertex_tag);
  const auto* end = vertex_by_tag.at(edge.end_vertex_tag);
  const double dr = end->r - begin->r;
  const double dz = end->z - begin->z;
  const double denominator = dr * dr + dz * dz;

  require(
      denominator > 0.0,
      "mesh_create",
      "zero-length geometric edge tag=" +
          std::to_string(edge.tag));

  const double parameter =
      ((point.r - begin->r) * dr +
       (point.z - begin->z) * dz) /
      denominator;

  require(
      std::isfinite(parameter),
      "mesh_create",
      "non-finite edge parameter for geometric edge tag=" +
          std::to_string(edge.tag));
  return parameter;
}

std::filesystem::path rank0_path(
    const std::filesystem::path& base) {
  return base.parent_path() /
      (base.stem().string() + "0" + base.extension().string());
}

void create_and_write_mesh(
    const Exchange& exchange,
    gmi_model* model) {
  apf::Mesh2* mesh = apf::makeEmptyMdsMesh(model, 2, false);
  require(
      mesh != nullptr,
      "mesh_create",
      "makeEmptyMdsMesh returned null");

  try {
    std::map<std::int32_t, const GeometricEdge*> edge_by_tag;
    for (const auto& edge : exchange.geometric_edges) {
      edge_by_tag.emplace(edge.tag, &edge);
    }

    std::vector<apf::MeshEntity*> mesh_vertices(
        exchange.nodes.size(), nullptr);

    for (std::size_t index = 0;
         index < exchange.nodes.size();
         ++index) {
      const auto& node = exchange.nodes[index];
      const auto& classification =
          exchange.vertex_classifications[index];

      apf::Vector3 coordinate(node.r, node.z, 0.0);
      apf::Vector3 parameter(0.0, 0.0, 0.0);

      if (classification.dimension == 1) {
        const auto* geometric_edge =
            edge_by_tag.at(classification.tag);
        parameter[0] =
            edge_parameter(exchange, *geometric_edge, node);
      } else if (classification.dimension == 2) {
        parameter[0] = node.r;
        parameter[1] = node.z;
      }

      mesh_vertices[index] = mesh->createVertex(
          find_model_entity(
              mesh,
              classification.dimension,
              classification.tag),
          coordinate,
          parameter);
      require(
          mesh_vertices[index] != nullptr,
          "mesh_create",
          "createVertex returned null for node=" +
              std::to_string(index));
    }

    std::map<
        std::pair<std::uint64_t, std::uint64_t>,
        apf::MeshEntity*>
        mesh_edge_entities;

    for (const auto& edge : exchange.mesh_edges) {
      apf::Downward downward;
      downward[0] =
          mesh_vertices[static_cast<std::size_t>(edge.node0)];
      downward[1] =
          mesh_vertices[static_cast<std::size_t>(edge.node1)];

      apf::MeshEntity* entity = mesh->createEntity(
          apf::Mesh::EDGE,
          find_model_entity(
              mesh,
              edge.classification.dimension,
              edge.classification.tag),
          downward);
      require(
          entity != nullptr,
          "mesh_create",
          "createEntity(EDGE) returned null");

      mesh_edge_entities.emplace(
          std::make_pair(edge.node0, edge.node1),
          entity);
    }

    for (std::size_t index = 0;
         index < exchange.triangles.size();
         ++index) {
      auto triangle_nodes = exchange.triangles[index].nodes;
      const double area = signed_double_area(
          exchange.nodes[static_cast<std::size_t>(triangle_nodes[0])],
          exchange.nodes[static_cast<std::size_t>(triangle_nodes[1])],
          exchange.nodes[static_cast<std::size_t>(triangle_nodes[2])]);

      if (area < 0.0) {
        std::swap(triangle_nodes[1], triangle_nodes[2]);
      }

      apf::Downward downward;
      downward[0] = mesh_edge_entities.at(
          edge_key(triangle_nodes[0], triangle_nodes[1]));
      downward[1] = mesh_edge_entities.at(
          edge_key(triangle_nodes[1], triangle_nodes[2]));
      downward[2] = mesh_edge_entities.at(
          edge_key(triangle_nodes[2], triangle_nodes[0]));

      const auto& classification =
          exchange.triangle_classifications[index];
      apf::MeshEntity* entity = mesh->createEntity(
          apf::Mesh::TRIANGLE,
          find_model_entity(
              mesh,
              classification.dimension,
              classification.tag),
          downward);
      require(
          entity != nullptr,
          "mesh_create",
          "createEntity(TRIANGLE) returned null for triangle=" +
              std::to_string(index));
    }

    mesh->acceptChanges();
    mesh->verify();
    mesh->writeNative(exchange.output_base.string().c_str());

    const auto output_rank0 = rank0_path(exchange.output_base);
    require(
        std::filesystem::is_regular_file(output_rank0),
        "write_native",
        "writeNative did not create rank-0 SMB: " +
            output_rank0.string());
    require(
        std::filesystem::file_size(output_rank0) > 0,
        "write_native",
        "writeNative created an empty rank-0 SMB: " +
            output_rank0.string());

    apf::disownMdsModel(mesh);
    mesh->destroyNative();
    apf::destroyMesh(mesh);
    mesh = nullptr;
  } catch (...) {
    if (mesh != nullptr) {
      apf::disownMdsModel(mesh);
      mesh->destroyNative();
      apf::destroyMesh(mesh);
    }
    throw;
  }
}

std::filesystem::path parse_arguments(int argc, char** argv) {
  if (argc != 3 || std::string(argv[1]) != "--input") {
    throw StageError(
        "arguments",
        "usage: linkmesh-scorec-helper --input <exchange-file>");
  }

  const std::filesystem::path input(argv[2]);
  require(
      std::filesystem::is_regular_file(input),
      "arguments",
      "exchange input is not a regular file: " + input.string());
  return input;
}

}  // namespace

int main(int argc, char** argv) {
  bool mpi_initialized = false;
  bool pcu_initialized = false;

  try {
    const auto input_path = parse_arguments(argc, argv);
    Exchange exchange = read_exchange(input_path);
    validate_exchange(exchange);

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
      throw StageError("mpi_init", "MPI_Init failed");
    }
    mpi_initialized = true;

    PCU_Comm_Init();
    pcu_initialized = true;

    int peer_count = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &peer_count);
    require(
        peer_count == 1 && PCU_Comm_Peers() == 1,
        "runtime_validate",
        "native SMB helper only supports one MPI rank");

    const auto output_rank0 = rank0_path(exchange.output_base);
    {
      NativeModel native_model;
      create_model(exchange, native_model);
      create_and_write_mesh(exchange, native_model.model);

      std::cout
          << "LINKMESH_M4_OK base="
          << percent_encode(exchange.output_base.string())
          << " rank0="
          << percent_encode(output_rank0.string())
          << '\n';
    }

    PCU_Comm_Free();
    pcu_initialized = false;

    MPI_Finalize();
    mpi_initialized = false;
    return 0;
  } catch (const StageError& error) {
    std::cerr
        << "LINKMESH_M4_ERROR stage="
        << percent_encode(error.stage())
        << " message="
        << percent_encode(single_line(error.what()))
        << '\n';
  } catch (const std::exception& error) {
    std::cerr
        << "LINKMESH_M4_ERROR stage=unexpected"
        << " message="
        << percent_encode(single_line(error.what()))
        << '\n';
  } catch (...) {
    std::cerr
        << "LINKMESH_M4_ERROR stage=unexpected"
        << " message=unknown_native_exception\n";
  }

  if (pcu_initialized) {
    PCU_Comm_Free();
  }
  if (mpi_initialized) {
    MPI_Finalize();
  }
  return 1;
}
