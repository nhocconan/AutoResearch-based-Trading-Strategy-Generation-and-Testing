# Strategy: 6h_Pivot_R2_S2_Breakout_1dEMA34_Volume_Spike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.066 | +17.7% | -14.9% | 150 | FAIL |
| ETHUSDT | 0.229 | +31.6% | -11.6% | 137 | PASS |
| SOLUSDT | 0.942 | +126.1% | -18.7% | 133 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.149 | +24.2% | -6.8% | 44 | PASS |
| SOLUSDT | 0.684 | +16.2% | -5.7% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot levels using previous day's HLC (no look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r2_1d = pp_1d + (high_1d - low_1d)  # R2 = PP + (High - Low)
    s2_1d = pp_1d - (high_1d - low_1d)  # S2 = PP - (High - Low)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe (primary timeframe)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period average on 6h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pp = pp_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        ema34 = ema34_aligned[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume + above EMA34
            if price > r2 and vol > 1.5 * vol_ma and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume + below EMA34
            elif price < s2 and vol > 1.5 * vol_ma and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through central pivot
            if position == 1 and price < pp:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Pivot_R2_S2_Breakout_1dEMA34_Volume_Spike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 04:21
