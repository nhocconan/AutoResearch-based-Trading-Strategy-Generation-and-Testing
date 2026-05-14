# Strategy: 4h_Camarilla_R4S4_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.191 | +29.2% | -9.3% | 140 | PASS |
| ETHUSDT | 0.149 | +27.6% | -13.1% | 139 | PASS |
| SOLUSDT | 0.863 | +125.8% | -17.2% | 113 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.873 | -3.2% | -8.2% | 52 | FAIL |
| ETHUSDT | 1.582 | +36.2% | -6.5% | 41 | PASS |
| SOLUSDT | 0.001 | +5.3% | -9.5% | 40 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla R4/S4 are stronger reversal/breakout levels than R3/S3, providing higher-probability
# signals with fewer false breakouts. Combined with 1d EMA34 for trend alignment and volume
# spike confirmation, this should reduce trade frequency while maintaining edge in both bull
# and bear markets by only taking trades in direction of daily trend. Target: 75-200 trades
# over 4 years (19-50/year) on 4h timeframe.

name = "4h_Camarilla_R4S4_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R4, S4)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = close + (high - low) * 1.1/2, S4 = close - (high - low) * 1.1/2
    camarilla_range = high_1d - low_1d
    r4 = close_1d + camarilla_range * 1.1 / 2
    s4 = close_1d - camarilla_range * 1.1 / 2
    
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 2.0x 20-period average (~3.3 days for 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for 1d EMA)
    start_idx = 34  # 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R4 with volume spike AND price > 1d EMA34 (bullish trend)
            if (close[i] > r4_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S4 with volume spike AND price < 1d EMA34 (bearish trend)
            elif (close[i] < s4_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S4 (breakdown) OR price below 1d EMA34 (trend change)
            if close[i] < s4_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above R4 (breakout) OR price above 1d EMA34 (trend change)
            if close[i] > r4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 09:14
