# Strategy: 4h_Pivot_R1_S1_Breakout_1dEMA34_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.602 | +53.5% | -6.7% | 212 | PASS |
| ETHUSDT | 0.299 | +36.1% | -10.3% | 171 | PASS |
| SOLUSDT | 0.368 | +39.6% | -14.1% | 85 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.193 | +3.6% | -7.5% | 77 | FAIL |
| ETHUSDT | 0.927 | +22.3% | -9.2% | 64 | PASS |
| SOLUSDT | 0.208 | +8.5% | -4.7% | 46 | PASS |

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
    
    # Calculate daily return for volatility filter
    daily_return = pd.Series(close_1d).pct_change().abs()
    vol_filter = pd.Series(daily_return).rolling(window=10, min_periods=10).mean().values
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vol_filter_val = vol_filter_aligned[i]
        pp = pp_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema34 = ema34_aligned[i]
        
        # Volatility filter: only trade in normal volatility (avoid chop)
        vol_condition = vol_filter_val < 0.03  # Less than 3% daily volatility
        
        # Volume spike: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + above EMA34 + normal vol
            if price > r1 and vol_spike and price > ema34 and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 + volume spike + below EMA34 + normal vol
            elif price < s1 and vol_spike and price < ema34 and vol_condition:
                signals[i] = -0.30
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through PP or volatility increases
            exit_signal = False
            
            if position == 1:  # long
                if price < pp or vol_filter_val > 0.05:  # High volatility exit
                    exit_signal = True
            elif position == -1:  # short
                if price > pp or vol_filter_val > 0.05:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_1dEMA34_Volume_Filter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 04:10
