# 2025-06-11: Strategy for 4h timeframe
# Hypothesis: Combining Bollinger Band squeeze with RSI momentum and volume confirmation
# captures breakouts from low volatility periods. Works in both bull and bear markets
# by trading the breakout direction only when volume confirms institutional interest.
# Uses 1-day timeframe for volatility regime filter to avoid false signals in ranging markets.

name = "4h_Bollinger_Squeeze_RSI_Momentum_Volume"
timeframe = "4h"
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

    # Get 1-day data for volatility regime and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Bollinger Bands (20, 2) on daily timeframe
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (std_20 * 2)
    lower_bb = sma_20 - (std_20 * 2)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width

    # Bollinger Squeeze: BB width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma

    # RSI (14) on daily timeframe
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    # Align daily indicators to 4h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bollinger squeeze breakout above upper band + RSI > 50 + volume spike
            if (squeeze_aligned[i] and 
                close[i] > upper_bb_aligned[i] and 
                rsi_aligned[i] > 50 and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bollinger squeeze breakout below lower band + RSI < 50 + volume spike
            elif (squeeze_aligned[i] and 
                  close[i] < lower_bb_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle of Bollinger Bands or squeeze ends
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if close[i] < middle_bb or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle of Bollinger Bands or squeeze ends
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if close[i] > middle_bb or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals