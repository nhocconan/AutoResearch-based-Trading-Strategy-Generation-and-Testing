# Strategy: 4h_CamarillaBreakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.163 | +9.0% | -17.8% | 103 | FAIL |
| ETHUSDT | 0.015 | +17.8% | -21.4% | 103 | PASS |
| SOLUSDT | 0.733 | +122.0% | -24.3% | 96 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.487 | +14.9% | -10.4% | 40 | PASS |
| SOLUSDT | -0.585 | -7.1% | -22.3% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with 12h Trend and Volume Spike
# - Uses Camarilla levels from daily timeframe (S1/S2 for long, R1/R2 for short)
# - Breakout above S1 with 12h uptrend or below R1 with 12h downtrend
# - Volume spike confirms breakout strength
# - Works in bull/bear by using 12h trend filter to avoid counter-trend trades
# - Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_CamarillaBreakout_12hTrend_Volume"
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
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data
    # S1 = C - (H-L)*1.08, S2 = C - (H-L)*1.16, R1 = C + (H-L)*1.08, R2 = C + (H-L)*1.16
    n1d = len(close_1d)
    camarilla_S1 = np.full(n1d, np.nan)
    camarilla_S2 = np.full(n1d, np.nan)
    camarilla_R1 = np.full(n1d, np.nan)
    camarilla_R2 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_val = H - L
        camarilla_S1[i] = C - range_val * 1.08
        camarilla_S2[i] = C - range_val * 1.16
        camarilla_R1[i] = C + range_val * 1.08
        camarilla_R2[i] = C + range_val * 1.16
    
    # Align Camarilla levels to 4h timeframe
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_S2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S2)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_R2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R2)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_S2_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_R2_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S1 (support) with 12h uptrend + volume spike
            long_cond = (close[i] > camarilla_S1_aligned[i] and 
                        ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below R1 (resistance) with 12h downtrend + volume spike
            short_cond = (close[i] < camarilla_R1_aligned[i] and 
                         ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S2 (strong support break)
            if close[i] < camarilla_S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R2 (strong resistance break)
            if close[i] > camarilla_R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 05:56
