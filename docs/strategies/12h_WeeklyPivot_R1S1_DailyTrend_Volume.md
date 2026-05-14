# Strategy: 12h_WeeklyPivot_R1S1_DailyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.294 | +3.9% | -16.8% | 54 | FAIL |
| ETHUSDT | 0.067 | +21.8% | -21.0% | 47 | PASS |
| SOLUSDT | 1.188 | +232.8% | -23.2% | 44 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.149 | +7.6% | -7.3% | 8 | PASS |
| SOLUSDT | -0.783 | -3.3% | -10.2% | 7 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Weekly Pivot (R1/S1) Breakout with Daily Trend Filter and Volume Spike
# Uses weekly Camarilla pivot levels for key support/resistance, daily EMA34 for trend alignment,
# and volume spike for confirmation. Designed for 12-37 trades/year to avoid fee drag.
# Works in bull markets (breakouts with trend) and bear markets (fades from pivot levels with trend).
name = "12h_WeeklyPivot_R1S1_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Get daily data for EMA trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Weekly Camarilla pivot levels (using previous week's OHLC)
    weekly_high = df_weekly['high'].shift(1).values
    weekly_low = df_weekly['low'].shift(1).values
    weekly_close = df_weekly['close'].shift(1).values
    
    # Camarilla calculations
    range_hl = weekly_high - weekly_low
    camarilla_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    camarilla_r1 = camarilla_pivot + range_hl * 1.1 / 12
    camarilla_s1 = camarilla_pivot - range_hl * 1.1 / 12
    camarilla_r2 = camarilla_pivot + range_hl * 1.1 / 6
    camarilla_s2 = camarilla_pivot - range_hl * 1.1 / 6
    camarilla_r3 = camarilla_pivot + range_hl * 1.1 / 4
    camarilla_s3 = camarilla_pivot - range_hl * 1.1 / 4
    camarilla_r4 = camarilla_pivot + range_hl * 1.1 / 2
    camarilla_s4 = camarilla_pivot - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 12h
    camarilla_pivot_12h = align_htf_to_ltf(prices, df_weekly, camarilla_pivot)
    camarilla_r1_12h = align_htf_to_ltf(prices, df_weekly, camarilla_r1)
    camarilla_s1_12h = align_htf_to_ltf(prices, df_weekly, camarilla_s1)
    camarilla_r2_12h = align_htf_to_ltf(prices, df_weekly, camarilla_r2)
    camarilla_s2_12h = align_htf_to_ltf(prices, df_weekly, camarilla_s2)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_weekly, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_weekly, camarilla_s3)
    camarilla_r4_12h = align_htf_to_ltf(prices, df_weekly, camarilla_r4)
    camarilla_s4_12h = align_htf_to_ltf(prices, df_weekly, camarilla_s4)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_12h[i]) or np.isnan(camarilla_pivot_12h[i]) or np.isnan(camarilla_r1_12h[i]) or 
            np.isnan(camarilla_s1_12h[i]) or np.isnan(camarilla_r2_12h[i]) or np.isnan(camarilla_s2_12h[i]) or 
            np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or np.isnan(camarilla_r4_12h[i]) or 
            np.isnan(camarilla_s4_12h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above R1 with daily uptrend and volume spike
            if close[i] > camarilla_r1_12h[i] and close[i] > ema34_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with daily downtrend and volume spike
            elif close[i] < camarilla_s1_12h[i] and close[i] < ema34_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below pivot OR daily trend turns down
            if close[i] < camarilla_pivot_12h[i] or close[i] < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above pivot OR daily trend turns up
            if close[i] > camarilla_pivot_12h[i] or close[i] > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 07:55
