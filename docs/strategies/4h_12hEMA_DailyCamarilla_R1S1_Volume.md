# Strategy: 4h_12hEMA_DailyCamarilla_R1S1_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.037 | +21.5% | -9.4% | 286 | PASS |
| ETHUSDT | 0.252 | +34.1% | -11.4% | 261 | PASS |
| SOLUSDT | 0.806 | +112.7% | -27.1% | 228 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.800 | -2.0% | -6.9% | 99 | FAIL |
| ETHUSDT | 0.817 | +19.4% | -7.4% | 93 | PASS |
| SOLUSDT | 0.126 | +7.3% | -11.0% | 74 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12-hour EMA Trend + Daily Camarilla Pivot R1/S1 Breakout with Volume Spike
# Uses 12h EMA for medium-term trend alignment and daily Camarilla pivot levels for institutional entry/exit.
# Volume spike confirms breakout strength. Designed for 15-25 trades/year to avoid fee drag.
# Works in bull markets (breakouts with trend) and bear markets (fades from pivot levels with trend).
name = "4h_12hEMA_DailyCamarilla_R1S1_Volume"
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
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Daily Camarilla pivot levels (using previous day's OHLC)
    daily_high = df_daily['high'].shift(1).values
    daily_low = df_daily['low'].shift(1).values
    daily_close = df_daily['close'].shift(1).values
    
    # Camarilla calculations
    range_hl = daily_high - daily_low
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3.0
    camarilla_r1 = camarilla_pivot + range_hl * 1.1 / 12
    camarilla_s1 = camarilla_pivot - range_hl * 1.1 / 12
    camarilla_r2 = camarilla_pivot + range_hl * 1.1 / 6
    camarilla_s2 = camarilla_pivot - range_hl * 1.1 / 6
    camarilla_r3 = camarilla_pivot + range_hl * 1.1 / 4
    camarilla_s3 = camarilla_pivot - range_hl * 1.1 / 4
    camarilla_r4 = camarilla_pivot + range_hl * 1.1 / 2
    camarilla_s4 = camarilla_pivot - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 4h
    camarilla_pivot_4h = align_htf_to_ltf(prices, df_daily, camarilla_pivot)
    camarilla_r1_4h = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_daily, camarilla_s1)
    camarilla_r2_4h = align_htf_to_ltf(prices, df_daily, camarilla_r2)
    camarilla_s2_4h = align_htf_to_ltf(prices, df_daily, camarilla_s2)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    camarilla_r4_4h = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    camarilla_s4_4h = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_4h[i]) or np.isnan(camarilla_pivot_4h[i]) or np.isnan(camarilla_r1_4h[i]) or 
            np.isnan(camarilla_s1_4h[i]) or np.isnan(camarilla_r2_4h[i]) or np.isnan(camarilla_s2_4h[i]) or 
            np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or np.isnan(camarilla_r4_4h[i]) or 
            np.isnan(camarilla_s4_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above R1 with 12h uptrend and volume spike
            if close[i] > camarilla_r1_4h[i] and close[i] > ema50_12h_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with 12h downtrend and volume spike
            elif close[i] < camarilla_s1_4h[i] and close[i] < ema50_12h_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below pivot OR 12h trend turns down
            if close[i] < camarilla_pivot_4h[i] or close[i] < ema50_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above pivot OR 12h trend turns up
            if close[i] > camarilla_pivot_4h[i] or close[i] > ema50_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 07:52
