import React, { useState } from "react";
import axios from "axios";
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import { Button, CircularProgress } from "@mui/material"; // Import CircularProgress
import { Table, TableContainer, TableHead, TableBody, TableRow, TableCell, Paper } from "@mui/material";

function App() {
  const [text, setText] = useState("");
  const [translations, setTranslations] = useState({});
  const [loading, setLoading] = useState(false); // State for loading

  const handleTranslate = async () => {
    setLoading(true); // Start loading
    try {
      const response = await axios.post("http://127.0.0.1:8000/translate", { text });
      setTranslations(response.data.translations);
    } catch (error) {
      console.error("Translation error:", error);
    } finally {
      setLoading(false); // Stop loading when request completes
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
        <h1 className="text-primary">English to Khowar Translator</h1>
      </div>

      <div>
        <p><strong>Khowar</strong> is an Indo-Aryan language spoken in northern Pakistan, particularly in the Chitral region.</p>
        <p><strong>Chitral (Khowar: ݯھیترار, romanized: ćhitrār)</strong> is a city situated on the Chitral River in the northern area of Khyber Pakhtunkhwa, Pakistan.</p>
      </div>

      {/* Input Box */}
      <Box
        component="form"
        sx={{
          input: { color: "white" }, // Text color
          label: { color: "white" }, // Label color
          "& .MuiOutlinedInput-root": {
            "& fieldset": { borderColor: "white" }, // Border color
            "&:hover fieldset": { borderColor: "lightgray" }, // Border on hover
            "&.Mui-focused fieldset": { borderColor: "white" }, // Border when focused
          },
        }}
        noValidate
        autoComplete="off"
      >
        <TextField
          id="outlined-basic"
          label="Input Text"
          variant="outlined"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to translate"
          style={{ marginRight: "10px", padding: "4px" }}
        />
      </Box>

      {/* Translate Button with Loading Indicator */}
      <div style={{ marginTop: '10px' }}>
        <Button
          sx={{
            color: "white",
            borderColor: "white",
            backgroundColor: "rgba(255, 255, 255, 0.2)",
            "&:hover": {
              backgroundColor: "rgba(255, 255, 255, 0.3)", // Hover effect
              borderColor: "lightgray",
            },
          }}
          variant="outlined"
          onClick={handleTranslate}
          disabled={loading} // Disable button while loading
          style={{ padding: "8px 15px", fontSize: "16px" }}
        >
          {loading ? <CircularProgress size={24} color="inherit" /> : "Translate"}
        </Button>
      </div>

      {/* Translation Table */}
      {Object.keys(translations).length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h2 style={{ textAlign: "center", color: "white" }}>Translation Results</h2>

          <TableContainer component={Paper} sx={{ width: "80%", margin: "auto", backgroundColor: "transparent", boxShadow: "none" }}>
            <Table sx={{ borderCollapse: "collapse" }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: "bold", fontSize: "1rem", color: "white", textAlign: "center", border: "1px solid white" }}>
                    Model
                  </TableCell>
                  <TableCell sx={{ fontWeight: "bold", fontSize: "1rem", color: "white", textAlign: "center", border: "1px solid white" }}>
                    Translation
                  </TableCell>
                </TableRow>
              </TableHead>

              <TableBody>
                {Object.entries(translations).map(([model, translation]) => (
                  <TableRow key={model} sx={{ backgroundColor: "transparent" }}>
                    <TableCell sx={{ color: "white", textAlign: "center", fontWeight: "bold", border: "1px solid white" }}>
                      {model}
                    </TableCell>
                    <TableCell sx={{ color: "white", textAlign: "center", border: "1px solid white" }}>
                      {translation}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </div>
      )}
    </div>
  );
}

export default App;
