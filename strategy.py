#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeFilter
Hypothesis: Camarilla pivot points on 1h provide precise entry/exit levels. 
Trend filter from 4h EMA20 ensures trades align with higher timeframe momentum.
Volume filter from 1d average volume > 1.5x 20-period average reduces false breakouts.
Designed for 15-30 trades/year on 1h to minimize fee drag while capturing trends in both bull and bear markets.
"""

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    # Calculate EMA20 on 4h close
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)

    # Get 1d data for volume filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    volume_1d = df_1d['volume'].values
    # Calculate 20-period average volume on 1d
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate Camarilla levels for 1h (using previous bar's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We calculate these for each bar using previous bar's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first bar values to avoid NaN propagation
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        vol_1d_current = volume_1d[i // 24] if i // 24 < len(volume_1d) else volume_1d[-1]
        vol_avg_val = vol_avg_20_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above R1 + 4h uptrend (close > EMA20) + 1d volume surge
            if (close[i] > R1[i] and 
                close[i] > ema_20_4h_aligned[i] and 
                vol_1d_current > vol_avg_val * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + 4h downtrend (close < EMA20) + 1d volume surge
            elif (close[i] < S1[i] and 
                  close[i] < ema_20_4h_aligned[i] and 
                  vol_1d_current > vol_avg_val * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or 4h trend turns down
            if (close[i] < S1[i] or close[i] < ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or 4h trend turns up
            if (close[i] > R1[i] or close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals