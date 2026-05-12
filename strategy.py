#!/usr/bin/env python3
# 1h_4H_Trend_Signal_1D_Volume_Confirmation
# Hypothesis: Use 4h trend direction (above/below EMA200) as signal direction, 
# confirmed by 1d volume spike (>1.5x average) and 1h RSI pullback (RSI<40 for long, RSI>60 for short).
# Enter on 1h pullback in direction of 4h trend with volume confirmation.
# Designed for low trade frequency (15-35/year) to avoid fee drag, works in bull/bear via trend filter.

name = "1h_4H_Trend_Signal_1D_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA200 trend filter
    ema_200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)

    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume average (20-day)
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)

    # 1h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_200_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend direction from 4h EMA200
        trend_up = close[i] > ema_200_4h_aligned[i]
        trend_down = close[i] < ema_200_4h_aligned[i]

        # Volume confirmation from 1d
        volume_spike = volume[i] > (1.5 * vol_avg_1d_aligned[i])

        # RSI conditions for entry
        rsi_oversold = rsi_values[i] < 40
        rsi_overbought = rsi_values[i] > 60

        if position == 0:
            # LONG: 4h uptrend AND volume spike AND RSI oversold
            if trend_up and volume_spike and rsi_oversold:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend AND volume spike AND RSI overbought
            elif trend_down and volume_spike and rsi_overbought:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h trend turns down OR RSI overbought (take profit)
            if not trend_up or rsi_values[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h trend turns up OR RSI oversold (take profit)
            if not trend_down or rsi_values[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals