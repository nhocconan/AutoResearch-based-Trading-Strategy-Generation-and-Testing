# Strategy: 6h_Camarilla_R3_S3_R4_S4_Breakout_Bounce_EMA34_Volume2x_1d

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.408 | +36.7% | -9.9% | 95 | PASS |
| ETHUSDT | 0.032 | +21.5% | -11.9% | 75 | PASS |
| SOLUSDT | 0.930 | +92.0% | -12.6% | 59 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.287 | +8.4% | -4.6% | 29 | PASS |
| ETHUSDT | 0.503 | +12.8% | -8.3% | 31 | PASS |
| SOLUSDT | 0.173 | +7.7% | -5.4% | 22 | PASS |

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
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivots from previous day's data
    # P = (H + L + C) / 3
    # Range = H - L
    # Resistance levels: R1 = P + Range * 1.1/12, R2 = P + Range * 1.1/6, R3 = P + Range * 1.1/4, R4 = P + Range * 1.1/2
    # Support levels: S1 = P - Range * 1.1/12, S2 = P - Range * 1.1/6, S3 = P - Range * 1.1/4, S4 = P - Range * 1.1/2
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    P = (prev_high + prev_low + prev_close) / 3
    Range = prev_high - prev_low
    
    R3 = P + Range * 1.1 / 4
    S3 = P - Range * 1.1 / 4
    R4 = P + Range * 1.1 / 2
    S4 = P - Range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or 
            np.isnan(ema34_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Break above R4 with uptrend (close > EMA34) and volume spike -> continuation
        # 2. Reject from S3 with uptrend and volume spike -> bounce
        long_breakout = (close[i] > R4_6h[i] and close[i-1] <= R4_6h[i-1] and 
                         close[i] > ema34_6h[i] and volume_spike[i])
        long_bounce = (close[i] > S3_6h[i] and close[i-1] <= S3_6h[i-1] and 
                       close[i] > ema34_6h[i] and volume_spike[i])
        
        # Short conditions:
        # 1. Break below S4 with downtrend (close < EMA34) and volume spike -> continuation
        # 2. Reject from R3 with downtrend and volume spike -> rejection
        short_breakout = (close[i] < S4_6h[i] and close[i-1] >= S4_6h[i-1] and 
                          close[i] < ema34_6h[i] and volume_spike[i])
        short_reject = (close[i] < R3_6h[i] and close[i-1] >= R3_6h[i-1] and 
                        close[i] < ema34_6h[i] and volume_spike[i])
        
        if long_breakout or long_bounce:
            signals[i] = 0.25
            position = 1
        elif short_breakout or short_reject:
            signals[i] = -0.25
            position = -1
        # Exit conditions: return to opposite S3/R3 level
        elif position == 1 and close[i] < S3_6h[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > R3_6h[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_R4_S4_Breakout_Bounce_EMA34_Volume2x_1d"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 21:24
