# Strategy: 4h_Pivot_R1_S1_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.533 | +40.0% | -5.7% | 293 | KEEP |
| ETHUSDT | 0.157 | +27.0% | -8.2% | 275 | KEEP |
| SOLUSDT | 0.459 | +53.8% | -17.4% | 229 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.751 | -5.9% | -6.7% | 113 | DISCARD |
| ETHUSDT | 0.831 | +15.8% | -6.2% | 107 | KEEP |
| SOLUSDT | 0.537 | +12.1% | -7.3% | 80 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    # Load 1d data once for Pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot levels (based on previous day's HLC)
    # Shift by 1 to avoid look-ahead: use previous day's data for today's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan  # First value invalid
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Pivot Point and support/resistance levels
    pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pp_1d - prev_low_1d
    s1_1d = 2 * pp_1d - prev_high_1d
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
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
        ema34 = ema34_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above R1 + volume spike + price > EMA34
            if price > r1 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 + volume spike + price < EMA34
            elif price < s1 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through PP or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below PP or volume dries up
                if price < pp or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above PP or volume dries up
                if price > pp or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 04:14
