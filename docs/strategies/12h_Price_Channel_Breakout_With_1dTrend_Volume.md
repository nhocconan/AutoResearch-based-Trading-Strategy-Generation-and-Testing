# Strategy: 12h_Price_Channel_Breakout_With_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.091 | +8.1% | -22.6% | 155 | FAIL |
| ETHUSDT | 0.012 | +11.5% | -22.7% | 193 | PASS |
| SOLUSDT | 0.737 | +165.9% | -41.2% | 170 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.505 | +18.3% | -12.8% | 55 | PASS |
| SOLUSDT | 0.423 | +16.5% | -16.7% | 53 | PASS |

## Code
```python
# 12h_Price_Channel_Breakout_With_1dTrend_Volume
# Hypothesis: Breakout of 20-period price channel on 12h timeframe with 1d trend filter and volume confirmation.
# Works in bull markets by capturing upward breakouts and in bear markets by capturing downward breakouts.
# Volume surge confirms institutional participation. Targets 12-37 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period price channel (Donchian) on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    price_channel_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    price_channel_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align higher timeframe data to 12h
    price_channel_high_aligned = align_htf_to_ltf(prices, df_12h, price_channel_high)
    price_channel_low_aligned = align_htf_to_ltf(prices, df_12h, price_channel_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(price_channel_high_aligned[i]) or np.isnan(price_channel_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > price_channel_high_aligned[i]
        breakout_down = close[i] < price_channel_low_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: upward breakout + uptrend + volume surge
        long_entry = breakout_up and trend_up and volume_surge[i]
        # Short: downward breakout + downtrend + volume surge
        short_entry = breakout_down and trend_down and volume_surge[i]
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Price_Channel_Breakout_With_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-28 05:41
