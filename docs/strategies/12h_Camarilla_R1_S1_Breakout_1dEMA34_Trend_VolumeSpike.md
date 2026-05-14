# Strategy: 12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.149 | +26.0% | -6.5% | 77 | PASS |
| ETHUSDT | 0.021 | +21.3% | -6.3% | 67 | PASS |
| SOLUSDT | 0.241 | +34.9% | -23.8% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.493 | -4.8% | -9.2% | 35 | FAIL |
| ETHUSDT | 0.292 | +9.1% | -4.0% | 27 | PASS |
| SOLUSDT | -0.964 | -5.4% | -15.1% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when price breaks above 1d Camarilla R1 AND price > 1d EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 1d Camarilla S1 AND price < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 1d Camarilla midpoint (mean reversion to median)
- Uses 1d EMA34 for trend alignment to avoid counter-trend trades
- Volume spike ensures institutional participation and reduces false breakouts
- Camarilla pivots provide clear structural levels that work in both bull and bear markets
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (based on previous day's range)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Midpoint = (R1 + S1)/2 = close (same as previous day's close)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12.0
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12.0
    camarilla_mid = prev_close  # Midpoint is previous day's close
    
    # Calculate EMA34 on 1d
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 35)  # Need 20 for volume MA, 35 for EMA34 (34+1 for shift)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r1_aligned[i]  # Break above Camarilla R1
        breakout_down = close[i] < camarilla_s1_aligned[i]  # Break below Camarilla S1
        
        # Trend filter
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Camarilla midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < camarilla_mid_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint
                if close[i] > camarilla_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 18:34
