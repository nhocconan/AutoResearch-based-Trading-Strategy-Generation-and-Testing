# Strategy: 4h_Camarilla_R1S1_Volume_Chop_Regime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.681 | +6.5% | -6.0% | 142 | FAIL |
| ETHUSDT | 0.109 | +24.5% | -6.4% | 132 | PASS |
| SOLUSDT | -0.508 | +2.8% | -12.0% | 109 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.604 | +11.6% | -3.8% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume spike and choppiness regime filter.
Long when price breaks above R1 (1d) AND 4h volume > 1.8x 20-bar average AND chop > 61.8 (range).
Short when price breaks below S1 (1d) AND 4h volume > 1.8x 20-bar average AND chop > 61.8 (range).
Exit when price touches 1d pivot point (PP) or opposite Camarilla level (S1 for long, R1 for short).
Uses 1d for Camarilla levels/PP and chop regime, 4h for execution and volume confirmation.
Designed to capture mean-reversion bounces in ranging markets with volume confirmation. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    # PP = (High + Low + Close)/3
    rng = high_1d - low_1d
    r1 = close_1d + 1.1 * rng / 12
    s1 = close_1d - 1.1 * rng / 12
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    # Simplified: use true range and rolling max/min
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr14 / (np.log10(14) * (max_high - min_low)))
    # Handle division by zero or invalid values
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # default to neutral chop
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x 20-bar average
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        ranging_market = chop_aligned[i] > 61.8
        
        # Breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Exit conditions: touch pivot or opposite level
        touch_pp = abs(close[i] - pp_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < s1_aligned[i]) or \
                         (position == -1 and close[i] > r1_aligned[i])
        
        if position == 0:
            # Long: break above R1 with volume confirmation and ranging market
            if (breakout_r1 and volume_confirmed and ranging_market):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and ranging market
            elif (breakout_s1 and volume_confirmed and ranging_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch pivot or break below S1
            if (touch_pp or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch pivot or break above R1
            if (touch_pp or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_Chop_Regime"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 18:28
