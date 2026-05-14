# Strategy: 4h_Camarilla_R1_S1_Breakout_12hEMA_Trend_Volume_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.005 | +20.0% | -10.9% | 207 | PASS |
| ETHUSDT | 0.483 | +51.3% | -13.4% | 198 | PASS |
| SOLUSDT | 0.706 | +97.3% | -27.2% | 168 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.396 | -8.1% | -9.3% | 77 | FAIL |
| ETHUSDT | 0.803 | +19.5% | -11.1% | 71 | PASS |
| SOLUSDT | -0.209 | +1.8% | -14.6% | 55 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hEMA_Trend_Volume_v4"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivot and levels from previous day's OHLC
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    prev_daily_range = prev_high_1d - prev_low_1d
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1 = pivot + 1.1 * prev_daily_range / 6
    s1 = pivot - 1.1 * prev_daily_range / 6
    
    # Align Camarilla levels to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(ema50_4h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average (balanced for BTC/ETH)
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above Camarilla R1 with uptrend and volume spike
            if close[i] > r1_4h[i] and close[i] > ema50_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 with downtrend and volume spike
            elif close[i] < s1_4h[i] and close[i] < ema50_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Camarilla S1 OR trend turns down
            if close[i] < s1_4h[i] or close[i] < ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Camarilla R1 OR trend turns up
            if close[i] > r1_4h[i] or close[i] > ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 08:21
