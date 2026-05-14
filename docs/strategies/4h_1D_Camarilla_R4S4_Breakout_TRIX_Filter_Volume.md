# Strategy: 4h_1D_Camarilla_R4S4_Breakout_TRIX_Filter_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.049 | +22.7% | -6.2% | 143 | PASS |
| ETHUSDT | 0.367 | +33.5% | -8.1% | 120 | PASS |
| SOLUSDT | 0.421 | +43.1% | -13.1% | 117 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.484 | -2.8% | -5.0% | 57 | FAIL |
| ETHUSDT | 0.713 | +13.8% | -6.8% | 54 | PASS |
| SOLUSDT | -0.102 | +4.9% | -6.4% | 43 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_1D_Camarilla_R4S4_Breakout_TRIX_Filter_Volume
# Hypothesis: Breakouts at daily Camarilla R4/S4 levels with TRIX momentum filter and volume confirmation.
# TRIX (triple EMA) filters out noise and confirms momentum direction. Works in bull/bear:
# - Buy when price breaks above R4 in bullish TRIX momentum with volume spike
# - Sell when price breaks below S4 in bearish TRIX momentum with volume spike
# Targets 20-50 trades/year on 4h timeframe to avoid fee drag. Focus on BTC/ETH.

name = "4h_1D_Camarilla_R4S4_Breakout_TRIX_Filter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for TRIX filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate TRIX (15-period triple EMA) on 1d close
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # First value has no previous
    trix_signal = trix  # Using TRIX itself as signal (positive = bullish momentum)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)

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

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_signal_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Momentum filter from TRIX
        bullish_momentum = trix_signal_aligned[i] > 0
        bearish_momentum = trix_signal_aligned[i] < 0

        if position == 0:
            # LONG: Break above Camarilla R4 in bullish momentum with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and bullish_momentum and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S4 in bearish momentum with volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and bearish_momentum and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R4 or momentum turns bearish
            if close[i] < camarilla_r4_aligned[i] or not bullish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S4 or momentum turns bullish
            if close[i] > camarilla_s4_aligned[i] or not bearish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 16:24
