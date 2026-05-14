# Strategy: 6h_DailyPivot_R1_S1_Breakout_Volume_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.102 | +24.5% | -10.4% | 232 | PASS |
| ETHUSDT | 0.453 | +41.5% | -9.4% | 203 | PASS |
| SOLUSDT | 0.516 | +61.9% | -16.1% | 197 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.014 | -9.1% | -9.7% | 84 | FAIL |
| ETHUSDT | 0.511 | +12.4% | -7.0% | 80 | PASS |
| SOLUSDT | -0.456 | -0.1% | -11.9% | 73 | FAIL |

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
    
    # Load daily data for pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's HLC for pivot calculation (no look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Daily pivot levels (standard formula)
    pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pp_1d - prev_low_1d  # R1 = 2*P - Low
    s1_1d = 2 * pp_1d - prev_high_1d  # S1 = 2*P - High
    r2_1d = pp_1d + (prev_high_1d - prev_low_1d)  # R2 = P + (High - Low)
    s2_1d = pp_1d - (prev_high_1d - prev_low_1d)  # S2 = P - (High - Low)
    
    # Daily ATR for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (14-period) on primary timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        atr_1d = atr_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume + volatility filter
            if price > r2 and vol > 1.5 * vol_ma and atr_1d > 0:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S2 with volume + volatility filter
            elif price < s2 and vol > 1.5 * vol_ma and atr_1d > 0:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions: mean reversion to R1/S1 or ATR stop
            # Mean reversion exit: price returns to R1 (for long) or S1 (for short)
            mean_rev_exit = (position == 1 and price < r1) or (position == -1 and price > s1)
            
            # ATR stop loss: 2.0 * ATR from entry
            stop_loss = (position == 1 and price < entry_price - 2.0 * atr_val) or \
                        (position == -1 and price > entry_price + 2.0 * atr_val)
            
            if mean_rev_exit or stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_DailyPivot_R1_S1_Breakout_Volume_ATRStop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 04:44
