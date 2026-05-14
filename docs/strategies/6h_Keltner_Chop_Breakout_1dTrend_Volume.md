# Strategy: 6h_Keltner_Chop_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.101 | +24.5% | -18.2% | 57 | PASS |
| ETHUSDT | 0.165 | +28.9% | -15.6% | 52 | PASS |
| SOLUSDT | 0.692 | +112.1% | -23.3% | 48 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.257 | -8.9% | -13.5% | 25 | FAIL |
| ETHUSDT | 0.448 | +14.2% | -9.8% | 21 | PASS |
| SOLUSDT | -0.331 | -2.6% | -21.0% | 20 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_Keltner_Chop_Breakout_1dTrend_Volume
# Hypothesis: 6s Keltner channel breakout with 1d trend filter and volume spike, filtered by Choppiness Index to avoid range-bound markets.
# Uses Keltner breakout for trend following, Choppiness Index > 61.8 to identify ranging conditions (avoid false breakouts),
# and volume spike for confirmation. Designed to work in both bull and bear markets by filtering counter-trend trades
# and avoiding whipsaws in low-volatility regimes.

name = "6h_Keltner_Chop_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')

    # Calculate Keltner Channel (20, 2) on 6h data
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean()
    upper_keltner = ema20 + (2 * atr)
    lower_keltner = ema20 - (2 * atr)

    # Calculate Choppiness Index (14) on 6h data
    atr14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10((atr14 * 14) / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # Neutral value when undefined

    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper Keltner with volume spike, uptrend, and not choppy
            if (close[i] > upper_keltner[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i] and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower Keltner with volume spike, downtrend, and not choppy
            elif (close[i] < lower_keltner[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Keltner or trend turns down
            if close[i] < lower_keltner[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Keltner or trend turns up
            if close[i] > upper_keltner[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 04:25
