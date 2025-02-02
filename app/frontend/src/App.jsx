import React, { useState } from "react";
import axios from "axios";
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';

function App() {
  const [text, setText] = useState("");
  const [translations, setTranslations] = useState({});

  const handleTranslate = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/translate", { text });
      setTranslations(response.data.translations);
    } catch (error) {
      console.error("Translation error:", error);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>English to Khowar Translator</h1>
      <div>
        <p>Khowar is an Indo-Aryan language spoken in northern Pakistan, particularly in the Chitral region</p>
      </div>
      <Box
      component="form"
      sx={{'& > :not(style)':{m:1, width:'25ch'}}}
      noValidate
      autocomplete="off"
      >
        <TextField id="outlined-basic" label="Outlined" variant="outlined" 
        value = {text}
        onChange = {(e) => setText(e.target.value)}
        placeholder="Enter text to translate"
        style = {{marginRight:"10px", padding:"5px"}}
        />

      </Box>
      <button onClick={handleTranslate} style={{ padding: "5px 10px" }}>Translate</button>

      {Object.keys(translations).length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h2>Translation Results:</h2>
          <table style={{ width: "80%", margin: "auto", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ border: "1px solid black", padding: "10px" }}>Model</th>
                <th style={{ border: "1px solid black", padding: "10px" }}>Translation</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(translations).map(([model, translation]) => (
                <tr key={model}>
                  <td style={{ border: "1px solid black", padding: "10px" }}><b>{model}</b></td>
                  <td style={{ border: "1px solid black", padding: "10px" }}>{translation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
