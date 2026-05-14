# Strategy: 6h_WilliamsFractal_Breakout_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.003 | +20.1% | -14.5% | 64 | PASS |
| ETHUSDT | 0.176 | +29.2% | -13.9% | 69 | PASS |
| SOLUSDT | 1.095 | +179.2% | -27.4% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.645 | +1.3% | -5.5% | 22 | FAIL |
| ETHUSDT | 0.303 | +9.6% | -9.6% | 19 | PASS |
| SOLUSDT | -0.249 | +1.9% | -9.5% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
# Uses Williams Fractal (1d) to identify potential reversal points, enters on breakout of the
# fractal level in the direction of the 1d trend (EMA50). Volume filter ensures breakout
# has participation. Designed for low trade frequency (~15-25/year) to minimize fee drag.
# Williams Fractals work well in ranging markets while trend filter avoids counter-trend trades.

name = "6h_WilliamsFractal_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractal and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Need 2 extra bars for fractal confirmation (Williams Fractal requires 2 bars after)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above bullish fractal with uptrend and volume
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bearish fractal with downtrend and volume
            elif (not np.isnan(bearish_fractal_aligned[i]) and 
                  close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA50 (trend change)
            if close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA50 (trend change)
            if close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 12:05
