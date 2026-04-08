# Strategy: 6h_ichimoku_1d_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.460 | -0.3% | -19.4% | 388 | DISCARD |
| ETHUSDT | -0.165 | +8.4% | -15.7% | 385 | DISCARD |
| SOLUSDT | 0.825 | +126.2% | -20.0% | 353 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.019 | +5.2% | -10.1% | 111 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter.
# Long when Tenkan > Kijun and price above Kumo (cloud) with bullish daily trend.
# Short when Tenkan < Kijun and price below Kumo with bearish daily trend.
# Uses daily trend filter to avoid counter-trend trades. Ichimoku provides clear trend signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "6h_ichimoku_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (high_series.rolling(window=9, min_periods=9).max() + 
              low_series.rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (high_series.rolling(window=26, min_periods=26).max() + 
             low_series.rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = ((high_series.rolling(window=52, min_periods=52).max() + 
                 low_series.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below Kijun or daily turn bearish
            if (close[i] < kijun[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Kijun or daily turn bullish
            if (close[i] > kijun[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with daily trend filter
            # Long: Tenkan > Kijun and price above Kumo during bullish day
            if (tenkan[i] > kijun[i] and 
                close[i] > senkou_a[i] and 
                close[i] > senkou_b[i] and 
                daily_bullish_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun and price below Kumo during bearish day
            elif (tenkan[i] < kijun[i] and 
                  close[i] < senkou_a[i] and 
                  close[i] < senkou_b[i] and 
                  daily_bearish_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
```

## Last Updated
2026-04-07 04:13
