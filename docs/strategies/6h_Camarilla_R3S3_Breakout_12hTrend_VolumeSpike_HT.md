# Strategy: 6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_HT

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.035 | +22.3% | -4.7% | 131 | KEEP |
| ETHUSDT | 0.349 | +34.7% | -6.8% | 116 | KEEP |
| SOLUSDT | 0.857 | +94.4% | -12.4% | 97 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.962 | -7.2% | -7.6% | 53 | DISCARD |
| ETHUSDT | 0.823 | +15.5% | -5.4% | 45 | KEEP |
| SOLUSDT | -1.103 | -6.4% | -13.9% | 38 | DISCARD |

## Code
```python
#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA25 trend filter
    ema_25_12h = pd.Series(df_12h['close'].values).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    r3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    s3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    
    # Align daily levels to 6h timeframe (with 1-day delay for completed bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 2.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours for 6h to reduce trades
    
    start_idx = max(200, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_25_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 12h trend direction
        trend_up = close > ema_25_12h_aligned[i]
        trend_down = close < ema_25_12h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R3 in uptrend with strong volume
            if (close[i] > r3_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S3 in downtrend with strong volume
            elif (close[i] < s3_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters Camarilla body (between R3 and S3) or trend change
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Camarilla body or trend change
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Using 6h timeframe with Camarilla R3/S3 breakouts, 12h EMA25 trend filter,
# and volume confirmation will yield 12-37 trades per year (50-150 total over 4 years).
# The strategy trades with the higher timeframe trend, capturing institutional breakouts
# in both bull and bear markets. Volume filter ensures breakouts have institutional
# participation. Position size of 0.25 manages drawdown, and cooldown of 2 bars
# prevents overtrading. This version uses 12h EMA25 as trend filter (vs 1d EMA34) to
# better capture medium-term trends and reduce whipsaw in choppy markets. The volume
# threshold is increased to 2.5x to ensure only high-conviction breakouts are taken.
```

## Last Updated
2026-05-07 12:28
