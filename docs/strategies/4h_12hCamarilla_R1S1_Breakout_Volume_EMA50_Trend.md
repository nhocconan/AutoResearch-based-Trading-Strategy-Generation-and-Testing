# Strategy: 4h_12hCamarilla_R1S1_Breakout_Volume_EMA50_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.158 | +16.3% | -7.1% | 317 | FAIL |
| ETHUSDT | 0.346 | +33.5% | -5.1% | 279 | PASS |
| SOLUSDT | 0.291 | +36.5% | -14.3% | 251 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.994 | +17.2% | -5.8% | 92 | PASS |
| SOLUSDT | 0.153 | +7.4% | -7.2% | 87 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Camarilla R1/S1 breakout + volume confirmation + 12h EMA50 trend filter.
Long when price breaks above 12h Camarilla R1 with volume confirmation and price > 12h EMA50 (uptrend).
Short when price breaks below 12h Camarilla S1 with volume confirmation and price < 12h EMA50 (downtrend).
Exit when price returns to the 12h Camarilla midpoint (H4/L4) or reverses with volume.
Uses 12h timeframe for structure (reduces noise) and 4h for entry timing and volume confirmation.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Camarilla levels provide precise support/resistance based on prior day's range, effective in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (based on prior 12h bar)
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.25*(high-low)
    # R2 = close + 1.166*(high-low)
    # R1 = close + 0.833*(high-low)
    # S1 = close - 0.833*(high-low)
    # S2 = close - 1.166*(high-low)
    # S3 = close - 1.25*(high-low)
    # S4 = close - 1.5*(high-low)
    # Midpoint H4/L4 = (R1 + S1) / 2 = close
    range_12h = high_12h - low_12h
    r1_12h = close_12h + 0.833 * range_12h
    s1_12h = close_12h - 0.833 * range_12h
    h4_12h = r1_12h  # Actually R1 is sometimes called H4
    l4_12h = s1_12h  # Actually S1 is sometimes called L4
    midpoint_12h = close_12h  # Camarilla midpoint is close
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    midpoint_12h_aligned = align_htf_to_ltf(prices, df_12h, midpoint_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(midpoint_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R1 with volume and uptrend (price > EMA50)
            if (close[i] > r1_12h_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S1 with volume and downtrend (price < EMA50)
            elif (close[i] < s1_12h_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below S1 with volume (reversal)
            if (close[i] <= midpoint_12h_aligned[i] or 
                (close[i] < s1_12h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above R1 with volume (reversal)
            if (close[i] >= midpoint_12h_aligned[i] or 
                (close[i] > r1_12h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hCamarilla_R1S1_Breakout_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 17:57
