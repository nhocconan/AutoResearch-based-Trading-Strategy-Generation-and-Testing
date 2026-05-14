# Strategy: 4h_1D_R4S4_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.482 | +39.3% | -7.9% | 282 | PASS |
| ETHUSDT | 0.073 | +23.3% | -9.3% | 280 | PASS |
| SOLUSDT | 0.306 | +40.9% | -22.0% | 276 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.401 | +3.0% | -5.6% | 94 | FAIL |
| ETHUSDT | 0.416 | +11.1% | -9.1% | 105 | PASS |
| SOLUSDT | 1.097 | +21.0% | -7.8% | 89 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_1D_R4S4_Breakout_1dTrend_Volume
# Hypothesis: Breakouts at daily Camarilla R4/S4 levels with 1d EMA trend filter and volume confirmation.
# R4/S4 represent strong support/resistance; breaks indicate momentum with institutional interest.
# Works in bull/bear: buy R4 breaks in uptrend, sell S4 breaks in downtrend. Volume confirms validity.
# Targets 20-50 trades/year on 4h timeframe to avoid fee drag. Focus on BTC/ETH.

name = "4h_1D_R4S4_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Get 1d data for Camarilla R4/S4 levels (from previous day)
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Camarilla R4 and S4 levels (outer bands)
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2

    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R4 in uptrend with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and uptrend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S4 in downtrend with volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and downtrend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R4 or trend reversal
            if close[i] < camarilla_r4_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S4 or trend reversal
            if close[i] > camarilla_s4_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 16:23
