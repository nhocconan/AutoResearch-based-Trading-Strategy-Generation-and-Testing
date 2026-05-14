# Strategy: 4h_KAMA_Trend_With_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.273 | -0.2% | -18.7% | 158 | FAIL |
| ETHUSDT | 0.178 | +30.1% | -24.4% | 146 | PASS |
| SOLUSDT | 0.770 | +149.1% | -37.8% | 152 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.269 | +10.4% | -11.6% | 55 | PASS |
| SOLUSDT | 0.881 | +28.2% | -9.1% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_With_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close for trend filter
    close_ser = pd.Series(df_1d['close'].values)
    change = abs(close_ser.diff(10))
    volatility = close_ser.diff(1).abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * 0.59 + 0.01) ** 2
    kama = [close_ser.iloc[0]]
    for i in range(1, len(close_ser)):
        kama.append(kama[-1] + sc.iloc[i] * (close_ser.iloc[i] - kama[-1]))
    kama = np.array(kama)
    
    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume spike detection on 4h (20-period MA)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Price above KAMA with volume spike
            if close[i] > kama_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA with volume spike
            elif close[i] < kama_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 10:33
