# Strategy: 6h_bb_squeeze_12h_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.339 | -7.0% | -7.5% | 64 | FAIL |
| ETHUSDT | -0.894 | -1.8% | -11.0% | 58 | FAIL |
| SOLUSDT | 0.103 | +24.6% | -12.4% | 60 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.078 | +6.7% | -4.6% | 19 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Bollinger Band Squeeze + 12h Trend Filter
# Hypothesis: During low volatility (BB width < 20th percentile), price breaks out in direction of 12h EMA(50) trend.
# Works in bull/bear by trading breakouts with trend filter. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_bb_squeeze_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma_20 + bb_std * std_20
    lower = sma_20 - bb_std * std_20
    bb_width = upper - lower
    
    # Bollinger Band width percentile (20-period lookback)
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(bb_width_pct[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width in lowest 20% of recent range
        squeeze = bb_width_pct[i] <= 0.20
        
        if position == 1:  # Long position
            # Exit: price closes below SMA(20) or trend changes
            if close[i] < sma_20[i] or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above SMA(20) or trend changes
            if close[i] > sma_20[i] or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if squeeze:
                # Breakout above upper band with uptrend
                if close[i] > upper[i] and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower band with downtrend
                elif close[i] < lower[i] and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 12:24
