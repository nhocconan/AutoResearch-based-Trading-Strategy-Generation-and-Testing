# Strategy: 4h_fractal_breakout_12h_trend_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.167 | +28.1% | -14.5% | 170 | KEEP |
| ETHUSDT | 0.436 | +49.2% | -15.9% | 161 | KEEP |
| SOLUSDT | 1.186 | +211.3% | -25.6% | 138 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.184 | -6.6% | -11.9% | 67 | DISCARD |
| ETHUSDT | 0.516 | +14.2% | -9.1% | 54 | KEEP |
| SOLUSDT | -0.268 | +0.1% | -14.2% | 49 | DISCARD |

## Code
```python
#!/usr/bin/env python3
# 4h_fractal_breakout_12h_trend_volume_v2
# Hypothesis: Williams Fractal breakouts on 4h with 12h EMA trend filter and volume confirmation.
# Uses 12h trend (slower than 1d) to reduce whipsaw in sideways markets while maintaining trend-following edge.
# Target: 25-40 trades/year to stay well under fee drag limits.

name = "4h_fractal_breakout_12h_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import align_htf_to_ltf, compute_williams_fractal_levels, get_htf_data

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 4h
    bearish_levels, bullish_levels = compute_williams_fractal_levels(high, low)
    bearish_fractal_level = pd.Series(
        align_htf_to_ltf(prices, prices, bearish_levels, additional_delay_bars=2)
    ).ffill().to_numpy()
    bullish_fractal_level = pd.Series(
        align_htf_to_ltf(prices, prices, bullish_levels, additional_delay_bars=2)
    ).ffill().to_numpy()
    
    # 12h EMA trend filter (34-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: volume > 2.0x 30-period average (~5 days)
    vol_period = 30
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(5, vol_period) + 5  # fractal needs 5 bars, plus volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_level[i]) or np.isnan(bullish_fractal_level[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below bullish fractal or trend fails
            if close[i] < bullish_fractal_level[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above bearish fractal or trend fails
            if close[i] > bearish_fractal_level[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Breakout long: price breaks above bullish fractal with uptrend
                if close[i] > bearish_fractal_level[i] and close[i] > ema_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below bearish fractal with downtrend
                elif close[i] < bullish_fractal_level[i] and close[i] < ema_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 10:56
