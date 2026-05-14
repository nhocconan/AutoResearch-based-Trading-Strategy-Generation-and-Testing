# Strategy: 12h_Camarilla_R1S1_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.226 | +29.2% | -6.4% | 96 | PASS |
| ETHUSDT | 0.060 | +22.7% | -8.0% | 86 | PASS |
| SOLUSDT | 0.029 | +17.8% | -22.5% | 88 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.006 | -2.2% | -7.0% | 41 | FAIL |
| ETHUSDT | 0.057 | +6.3% | -6.8% | 33 | PASS |
| SOLUSDT | -0.678 | -3.2% | -15.4% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with Volume Spike and 1d Trend Filter
Hypothesis: Camarilla pivot levels (R1/S1) derived from 1d range act as strong support/resistance.
Breaks of these levels with volume confirmation and aligned with 1d EMA trend capture
trend continuation moves. Works in bull/bear by only taking breakouts in trend direction.
Low frequency (~15-25/year) minimizes fee drag while capturing explosive moves.
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
    
    # Get 1d data for Camarilla pivots and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are close, high, low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.8x 20-period volume average (on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        trend = ema34_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Look for breakout of R1/S1 with volume, in trend direction
            if vol_ok:
                # Break above R1 in uptrend
                if price > r1 and price > trend:
                    signals[i] = 0.25
                    position = 1
                # Break below S1 in downtrend
                elif price < s1 and price < trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if price returns to S1 or trend reverses
            if price < s1 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to R1 or trend reverses
            if price > r1 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-18 06:10
