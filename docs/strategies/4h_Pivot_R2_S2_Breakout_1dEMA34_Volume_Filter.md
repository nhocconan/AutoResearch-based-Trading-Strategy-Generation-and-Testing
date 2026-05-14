# Strategy: 4h_Pivot_R2_S2_Breakout_1dEMA34_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.399 | +36.7% | -7.3% | 157 | PASS |
| ETHUSDT | 0.029 | +21.6% | -8.0% | 130 | PASS |
| SOLUSDT | -0.123 | +16.1% | -13.4% | 59 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.890 | -1.6% | -6.4% | 61 | FAIL |
| ETHUSDT | 0.757 | +16.2% | -5.1% | 48 | PASS |

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
    
    # Load daily data once
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
    
    # Daily volatility filter (10-day average of absolute returns)
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
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
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
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        ema34 = ema34_aligned[i]
        
        # Volatility filter: only trade in normal volatility (avoid chop)
        vol_condition = vol_filter_val < 0.03  # Less than 3% daily volatility
        
        # Volume spike: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above R2 + volume spike + above EMA34 + normal vol
            if price > r2 and vol_spike and price > ema34 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 + volume spike + below EMA34 + normal vol
            elif price < s2 and vol_spike and price < ema34 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through central pivot or volatility increases
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Pivot_R2_S2_Breakout_1dEMA34_Volume_Filter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 04:14
