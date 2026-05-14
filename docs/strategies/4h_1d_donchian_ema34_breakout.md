# Strategy: 4h_1d_donchian_ema34_breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.191 | +31.1% | -19.3% | 87 | KEEP |
| ETHUSDT | 0.172 | +29.2% | -21.3% | 84 | KEEP |
| SOLUSDT | 0.661 | +132.2% | -41.0% | 92 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.884 | -7.4% | -11.3% | 31 | DISCARD |
| ETHUSDT | 0.678 | +22.2% | -8.8% | 24 | KEEP |
| SOLUSDT | 0.360 | +13.7% | -11.9% | 27 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 10-period Donchian channels on daily
    high_10 = np.full(len(close_1d), np.nan)
    low_10 = np.full(len(close_1d), np.nan)
    for i in range(10, len(close_1d)):
        high_10[i] = np.max(high_1d[i-10:i])
        low_10[i] = np.min(low_1d[i-10:i])
    
    # Calculate 34-period EMA on daily (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 4h timeframe
    high_10_aligned = align_htf_to_ltf(prices, df_1d, high_10)
    low_10_aligned = align_htf_to_ltf(prices, df_1d, low_10)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.30  # 30% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(high_10_aligned[i]) or 
            np.isnan(low_10_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA34
        above_ema = close[i] > ema_34_aligned[i]
        below_ema = close[i] < ema_34_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_10_aligned[i]
        short_breakout = close[i] < low_10_aligned[i]
        
        # Entry conditions: breakout in direction of trend
        long_entry = long_breakout and above_ema
        short_entry = short_breakout and below_ema
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (short_breakout or below_ema)
        exit_short = position == -1 and (long_breakout or above_ema)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_ema34_breakout"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 13:04
