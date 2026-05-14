# Strategy: 4h_Camarilla_R3_S3_Reversal_12hEMA50_Volume_Session_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.217 | +15.1% | -9.7% | 379 | DISCARD |
| ETHUSDT | 0.464 | +38.0% | -9.0% | 343 | KEEP |
| SOLUSDT | 0.481 | +47.9% | -16.1% | 272 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.459 | +10.7% | -5.0% | 132 | KEEP |
| SOLUSDT | -0.107 | +4.8% | -9.4% | 100 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour Camarilla pivot reversal with 12-hour EMA50 trend filter and volume spike
    # Works in bull/bear via trend filter: only take long in uptrend, short in downtrend.
    # Camarilla reversals capture mean reversion at key levels; EMA50 filters trend; volume confirms.
    # Targets ~25 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Camarilla pivot calculation (use previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # Need daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # We'll use R3/S3 as primary levels
    
    # Calculate daily range from previous day (to avoid look-ahead)
    # For each 1d bar, use previous day's OHLC
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First bar will have NaN due to roll, handled by min_periods later
    
    # Calculate Camarilla levels for each 1d bar using previous day's data
    daily_range = prev_high_1d - prev_low_1d
    # Avoid division by zero
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)
    
    # Camarilla R3 and S3 levels
    r3_1d = prev_close_1d + daily_range * 1.1 / 4
    s3_1d = prev_close_1d - daily_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above S3 (support) with volume + price above 12h EMA50 (uptrend)
            if close[i] > s3_1d_aligned[i] and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below R3 (resistance) with volume + price below 12h EMA50 (downtrend)
            elif close[i] < r3_1d_aligned[i] and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Camarilla level or trend reversal vs 12h EMA50
            if position == 1:
                if close[i] < r3_1d_aligned[i] or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > s3_1d_aligned[i] or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Reversal_12hEMA50_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 08:16
