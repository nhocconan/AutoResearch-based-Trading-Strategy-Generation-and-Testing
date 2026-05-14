# Strategy: 1h_Pivot_S1R1_Volume_4hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.373 | +5.0% | -10.5% | 977 | FAIL |
| ETHUSDT | 0.104 | +24.7% | -10.4% | 917 | PASS |
| SOLUSDT | 0.596 | +79.0% | -21.8% | 889 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.832 | +18.2% | -7.7% | 300 | PASS |
| SOLUSDT | 0.116 | +7.1% | -8.3% | 287 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d for signal direction, 1h for entry timing.
# Uses daily pivot points for structure, volume filter for confirmation, and 4h EMA for trend.
# Targets 15-37 trades/year by requiring confluence of multiple filters.
# Works in bull/bear: uses price relative to pivot (support/resistance) and EMA trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day OHLC)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    
    # Align daily pivots to 1h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily trend: price above/below daily EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume filter: volume > 1.5 x 20-period average (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need pivots (1), daily EMA (34), volume MA (20), 4h EMA (20)
    start_idx = max(1, 34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters
        daily_bullish = price > ema_34_1d_aligned[i]
        daily_bearish = price < ema_34_1d_aligned[i]
        fourh_bullish = price > ema_20_4h_aligned[i]
        fourh_bearish = price < ema_20_4h_aligned[i]
        
        if position == 0:
            # Long: price crosses above S1 with volume, daily bullish, and 4h bullish
            if price > s1_aligned[i] and vol_filter and daily_bullish and fourh_bullish:
                signals[i] = size
                position = 1
            # Short: price crosses below R1 with volume, daily bearish, and 4h bearish
            elif price < r1_aligned[i] and vol_filter and daily_bearish and fourh_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below pivot or daily/4h trend turns bearish
            if price < pivot_aligned[i] or not daily_bullish or not fourh_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above pivot or daily/4h trend turns bullish
            if price > pivot_aligned[i] or not daily_bearish or not fourh_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Pivot_S1R1_Volume_4hTrend"
timeframe = "1h"
leverage = 1.0
```

## Last Updated
2026-04-27 10:25
