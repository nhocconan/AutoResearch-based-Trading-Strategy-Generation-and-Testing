# Strategy: 6h_DailyPivot_R1S2_VolumeTrendFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.485 | +32.7% | -3.3% | 207 | PASS |
| ETHUSDT | 0.435 | +32.4% | -4.3% | 176 | PASS |
| SOLUSDT | -0.363 | +8.7% | -15.4% | 168 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.717 | -1.3% | -2.9% | 79 | FAIL |
| ETHUSDT | 0.277 | +8.3% | -3.9% | 82 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily pivot points with volume confirmation and trend filter
# Daily pivots (R1/S1 for breakouts, R2/S2 for reversals) provide key intraday levels
# Breakout above R1 or below S1 with volume > 1.8x 20-period average indicates strong momentum
# Rejection at R2 or S2 with volume confirmation indicates mean reversion within daily range
# Trend filter: 20-period EMA on 6h timeframe to avoid counter-trend trades
# Works in bull/bear markets: breakouts capture trends, reversals capture pullbacks within trend
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_DailyPivot_R1S2_VolumeTrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point calculation
    # Pivot = (previous high + previous low + previous close) / 3
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Support and Resistance levels
    r1 = pivot + (range_ * 1.0)
    r2 = pivot + (range_ * 2.0)
    s1 = pivot - (range_ * 1.0)
    s2 = pivot - (range_ * 2.0)
    
    # Align daily levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: >1.8x 20-period average (higher threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Trend filter: 20-period EMA on 6h timeframe
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend = close > ema_20
    downtrend = close < ema_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(ema_20[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume confirmation and uptrend
            if close[i] > r1_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume confirmation and downtrend
            elif close[i] < s1_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects S2 with volume confirmation (bounce from support)
            elif close[i] < s2_aligned[i] and close[i] > s2_aligned[i] * 0.995 and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R2 with volume confirmation (rejection from resistance)
            elif close[i] > r2_aligned[i] and close[i] < r2_aligned[i] * 1.005 and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (failed support) or reaches R2 (take profit)
            if close[i] < s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (failed resistance) or reaches S2 (take profit)
            if close[i] > r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 21:50
