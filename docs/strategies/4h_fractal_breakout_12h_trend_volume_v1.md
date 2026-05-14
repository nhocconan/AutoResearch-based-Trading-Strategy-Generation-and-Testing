# Strategy: 4h_fractal_breakout_12h_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.005 | +18.6% | -19.7% | 134 | KEEP |
| ETHUSDT | 0.163 | +28.8% | -18.3% | 127 | KEEP |
| SOLUSDT | 0.941 | +180.0% | -29.6% | 129 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.557 | -0.7% | -6.9% | 44 | DISCARD |
| ETHUSDT | 0.405 | +12.8% | -9.8% | 39 | KEEP |
| SOLUSDT | 0.528 | +16.5% | -10.6% | 42 | KEEP |

## Code
```python
# 4h_fractal_breakout_12h_trend_volume_v1
# Hypothesis: Williams Fractal breakout on 4h with 12h EMA trend filter and volume confirmation.
# Long when price breaks above recent bearish fractal (resistance) in uptrend with volume.
# Short when price breaks below recent bullish fractal (support) in downtrend with volume.
# Williams fractals require 2-bar confirmation, so we use additional_delay_bars=2.
# Timeframe: 4h, HTF: 12h for trend filter.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_fractal_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import align_htf_to_ltf, compute_williams_fractal_levels, get_htf_data

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 12h timeframe (more reliable than 1d for 4h strategy)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 12h data
    bearish_levels, bullish_levels = compute_williams_fractal_levels(
        df_12h['high'].values,
        df_12h['low'].values,
    )
    
    # Align fractals to 4h timeframe with 2-bar confirmation delay
    bearish_fractal_level = pd.Series(
        align_htf_to_ltf(prices, df_12h, bearish_levels, additional_delay_bars=2)
    ).ffill().to_numpy()
    bullish_fractal_level = pd.Series(
        align_htf_to_ltf(prices, df_12h, bullish_levels, additional_delay_bars=2)
    ).ffill().to_numpy()
    
    # 12h EMA trend filter (50)
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(20, 50) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_level[i]) or np.isnan(bullish_fractal_level[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below bullish fractal (support) or trend fails
            if close[i] < bullish_fractal_level[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above bearish fractal (resistance) or trend fails
            if close[i] > bearish_fractal_level[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above bearish fractal (resistance) with uptrend and volume
            if close[i] > bearish_fractal_level[i] and close[i] > ema_12h_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below bullish fractal (support) with downtrend and volume
            elif close[i] < bullish_fractal_level[i] and close[i] < ema_12h_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 10:56
