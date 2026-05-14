# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeS_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.126 | +24.7% | -7.4% | 175 | PASS |
| ETHUSDT | 0.050 | +22.7% | -4.2% | 153 | PASS |
| SOLUSDT | 0.558 | +55.6% | -10.9% | 133 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.488 | +3.3% | -4.7% | 63 | FAIL |
| ETHUSDT | 0.372 | +9.5% | -6.1% | 62 | PASS |
| SOLUSDT | 0.172 | +7.6% | -4.4% | 51 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeS_v4
# Hypothesis: Refines prior version by tightening entry conditions to reduce trade frequency.
# Uses 1-day EMA34 trend filter, 4h volume spike (30-period average), and requires price to close
# beyond R1/S1 levels. Adds minimum holding period of 3 bars to prevent whipsaw exits.
# Target: 20-30 trades/year to minimize fee drag while maintaining edge in both bull/bear.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeS_v4"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Get 1d data for trend filter (EMA34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter on 4h (30-period average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.5 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > R1, above 1d EMA34 trend, volume spike
            if close[i] > r1_4h[i] and close[i] > ema_34_1d_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < S1, below 1d EMA34 trend, volume spike
            elif close[i] < s1_4h[i] and close[i] < ema_34_1d_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit conditions: require minimum 3 bars held
            if bars_since_entry >= 3:
                if close[i] < r1_4h[i] or close[i] < ema_34_1d_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:
                # Hold position for minimum period
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: require minimum 3 bars held
            if bars_since_entry >= 3:
                if close[i] > s1_4h[i] or close[i] > ema_34_1d_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
            else:
                # Hold position for minimum period
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 00:34
