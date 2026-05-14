# Strategy: 6h_Williams_Fractal_Breakout_12hEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.020 | +21.3% | -13.1% | 66 | PASS |
| ETHUSDT | 0.359 | +37.8% | -9.1% | 50 | PASS |
| SOLUSDT | 1.386 | +212.8% | -12.9% | 54 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.226 | -2.4% | -6.5% | 27 | FAIL |
| ETHUSDT | 0.206 | +8.2% | -6.9% | 22 | PASS |
| SOLUSDT | -1.079 | -6.9% | -13.6% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams Fractal breakout with 12-hour trend filter and volume confirmation.
Trades breakouts above/below confirmed Williams fractals in the direction of the 12h EMA trend.
Williams fractals require 2-bar confirmation (higher timeframe) to avoid false breaks.
Targets 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.
Works in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams fractals and trend - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 12h (requires 5-bar window: 2 left, center, 2 right)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_12h, low_12h)
    
    # Williams fractals need 2 extra 12h bars for confirmation (beyond the close wait)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # 12h EMA for trend filter (34-period)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above bullish fractal (resistance), above 12h EMA (uptrend)
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal (support), below 12h EMA (downtrend)
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite fractal level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches bearish fractal (support) or closes below 12h EMA
                if close[i] < bearish_fractal_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches bullish fractal (resistance) or closes above 12h EMA
                if close[i] > bullish_fractal_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 18:33
