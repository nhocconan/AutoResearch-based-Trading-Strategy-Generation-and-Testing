# Strategy: 6h_WilliamsR_MeanReversion_1dEMA34_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.089 | +23.9% | -8.8% | 160 | PASS |
| ETHUSDT | -0.422 | +5.0% | -10.7% | 155 | FAIL |
| SOLUSDT | -0.019 | +18.1% | -15.2% | 153 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.245 | +7.7% | -2.8% | 62 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h Williams %R Mean Reversion with 1d EMA34 Trend Filter
Hypothesis: Williams %R identifies overbought/oversold conditions on 6h chart.
In strong 1d trends (EMA34), extreme %R readings often precede mean-reversion pullbacks.
Long when %R < -80 (oversold) and price > 1d EMA34 (uptrend).
Short when %R > -20 (overbought) and price < 1d EMA34 (downtrend).
Uses discrete sizing (0.25) to minimize fee churn. Targets 50-150 total trades over 4 years.
Works in both bull and bear markets: trend filter ensures we trade with higher timeframe momentum,
while %R extremes provide timely entry points for pullbacks in the direction of the 1d trend.
"""

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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R and EMA
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_trend = ema_34_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Williams %R oversold (< -80) AND price > 1d EMA34 (uptrend)
            long_entry = (wr < -80.0) and (curr_close > ema_trend)
            # Short: Williams %R overbought (> -20) AND price < 1d EMA34 (downtrend)
            short_entry = (wr > -20.0) and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Williams %R rises above -50 (momentum fading) OR price crosses below EMA
            if (wr > -50.0) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Williams %R falls below -50 (momentum fading) OR price crosses above EMA
            if (wr < -50.0) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_1dEMA34_Trend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 05:17
