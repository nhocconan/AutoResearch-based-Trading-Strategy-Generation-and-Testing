# Strategy: 6h_WeeklyPivot_DailyBreakout_TrendFilter_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.341 | +40.7% | -13.3% | 63 | PASS |
| ETHUSDT | 0.117 | +25.1% | -15.1% | 66 | PASS |
| SOLUSDT | 0.456 | +69.2% | -37.1% | 71 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.909 | -0.3% | -5.5% | 18 | FAIL |
| ETHUSDT | 0.232 | +9.3% | -8.0% | 21 | PASS |
| SOLUSDT | -0.421 | -3.3% | -15.2% | 20 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyBreakout_TrendFilter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (pivot based on prior week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Range = H - L
    rng = weekly_high - weekly_low
    # Resistance 3 = H + 2*(PP - L)
    r3 = weekly_high + 2 * (pp - weekly_low)
    # Support 3 = L - 2*(H - PP)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivot to 6h timeframe (with 1-bar delay for completed weekly bar)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: spike above 2.0x 24-period average (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_6h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above weekly S3, daily uptrend (price > EMA34), volume breakout
            if (close[i] > s3_6h[i] and 
                close[i] > ema_34_6h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly R3, daily downtrend (price < EMA34), volume breakdown
            elif (close[i] < r3_6h[i] and 
                  close[i] < ema_34_6h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily S3 or trend reversal
            if close[i] < s3_6h[i] or close[i] < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly R3 or trend reversal
            if close[i] > r3_6h[i] or close[i] > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 12:06
