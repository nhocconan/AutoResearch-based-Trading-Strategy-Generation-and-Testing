#!/usr/bin/env python3
# 1h_4H_1D_RSI_Trend_Reversal
# Hypothesis: Use 4h RSI for mean-reversion signals in 1h timeframe, filtered by 1d EMA200 trend and volume spike.
# Long when 4h RSI < 30 (oversold) + price > 1d EMA200 (uptrend) + volume spike.
# Short when 4h RSI > 70 (overbought) + price < 1d EMA200 (downtrend) + volume spike.
# Exit when RSI returns to neutral (40-60 range) or trend changes.
# Designed for low trade frequency (15-37/year) with mean-reversion edge in both bull and bear markets.

name = "1h_4H_1D_RSI_Trend_Reversal"
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

    # Get 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI(14)
    close_4h = df_4h['close']
    delta = close_4h.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_values = rsi_4h.values
    
    # Align 4h RSI to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)

    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h RSI < 30 (oversold) + price > 1d EMA200 (uptrend) + volume spike
            if (rsi_4h_aligned[i] < 30 and 
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h RSI > 70 (overbought) + price < 1d EMA200 (downtrend) + volume spike
            elif (rsi_4h_aligned[i] > 70 and 
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (40-60) or trend changes (price below EMA200)
            if (rsi_4h_aligned[i] >= 40 and rsi_4h_aligned[i] <= 60) or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (40-60) or trend changes (price above EMA200)
            if (rsi_4h_aligned[i] >= 40 and rsi_4h_aligned[i] <= 60) or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals