# Strategy: 4h_Williams_Fractal_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.547 | +47.7% | -10.0% | 56 | PASS |
| ETHUSDT | 0.479 | +48.4% | -9.5% | 55 | PASS |
| SOLUSDT | 1.048 | +165.8% | -23.3% | 65 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.239 | -2.5% | -6.8% | 28 | FAIL |
| ETHUSDT | 0.785 | +15.7% | -5.8% | 15 | PASS |
| SOLUSDT | 0.291 | +9.7% | -6.5% | 16 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Williams_Fractal_Breakout_1dTrend_Volume
# Hypothesis: Williams Fractals from 1d chart identify key swing highs/lows. 
# Break above recent bearish fractal with 1d uptrend and volume confirmation = long.
# Break below recent bullish fractal with 1d downtrend and volume confirmation = short.
# Exit on opposite fractal or trend reversal. Designed for low-frequency, high-conviction trades.

name = "4h_Williams_Fractal_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Fractals: bearish (sell) fractal = high with lower highs on both sides
    # bullish (buy) fractal = low with higher lows on both sides
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Williams fractals require 2-bar confirmation after center bar
    bearish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 1.8 * 20-period average (higher threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(bearish_fractal_confirmed[i]) or 
            np.isnan(bullish_fractal_confirmed[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        price_above_bearish = close[i] > bearish_fractal_confirmed[i]
        price_below_bullish = close[i] < bullish_fractal_confirmed[i]
        
        # Trend conditions
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above bearish fractal + uptrend + volume spike
            if price_above_bearish and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bullish fractal + downtrend + volume spike
            elif price_below_bullish and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below bullish fractal OR trend reversal
            if price_below_bullish or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above bearish fractal OR trend reversal
            if price_above_bearish or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 02:21
