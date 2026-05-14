# Strategy: 4h_Pivot_R2_S2_Breakout_1dEMA34_Volume_Spike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.227 | +31.5% | -8.9% | 177 | PASS |
| ETHUSDT | 0.053 | +21.2% | -15.2% | 173 | PASS |
| SOLUSDT | 0.755 | +117.0% | -21.4% | 164 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.836 | -2.9% | -8.5% | 63 | FAIL |
| ETHUSDT | 0.457 | +13.4% | -9.3% | 59 | PASS |
| SOLUSDT | -0.439 | -2.8% | -13.1% | 56 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    r1_1d = 2 * pp_1d - prev_low_1d
    s1_1d = 2 * pp_1d - prev_high_1d
    r2_1d = pp_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pp_1d - (prev_high_1d - prev_low_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe (primary timeframe)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period average on 4h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
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
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        ema34 = ema34_aligned[i]
        
        if position == 0:
            # Long: price breaks above R2 + volume spike + above EMA34
            if price > r2 and vol > 1.5 * vol_ma and price > ema34:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S2 + volume spike + below EMA34
            elif price < s2 and vol > 1.5 * vol_ma and price < ema34:
                signals[i] = -0.30
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
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Pivot_R2_S2_Breakout_1dEMA34_Volume_Spike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 04:16
