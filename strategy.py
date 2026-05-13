#!/usr/bin/env python3
# 4h_12h_RSI_Reversal_With_Volume_Confirmation
# Hypothesis: Use RSI divergence on 12h timeframe for mean reversion signals in ranging markets.
# Long when 12h RSI < 30 and price touches 4h Bollinger Lower Band with volume spike.
# Short when 12h RSI > 70 and price touches 4h Bollinger Upper Band with volume spike.
# Exit when price crosses 4h SMA20 or RSI returns to neutral zone (40-60).
# Designed for low turnover (~20-30/year) to avoid fee drag in both bull and bear markets.
# Uses 12h RSI for regime filtering and 4h Bollinger Bands for precise entry timing.

name = "4h_12h_RSI_Reversal_With_Volume_Confirmation"
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

    # Get 12h data for RSI calculation (primary filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h RSI(14)
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_12h = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    rsi_12h[:13] = np.nan  # Not enough data
    
    # Align 12h RSI to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)

    # 4h Bollinger Bands (20, 2)
    sma_20 = np.full(n, np.nan)
    std_20 = np.full(n, np.nan)
    for i in range(19, n):
        sma_20[i] = np.mean(close[i-19:i+1])
        std_20[i] = np.std(close[i-19:i+1])
    
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    sma_20_for_exit = sma_20.copy()  # For exit condition

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(lower_band[i]) or 
            np.isnan(upper_band[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(sma_20_for_exit[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold + touches lower band + volume spike
            if (rsi_12h_aligned[i] < 30 and 
                close[i] <= lower_band[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought + touches upper band + volume spike
            elif (rsi_12h_aligned[i] > 70 and 
                  close[i] >= upper_band[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses above SMA20 OR RSI returns to neutral
            if (close[i] >= sma_20_for_exit[i] or 
                rsi_12h_aligned[i] >= 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below SMA20 OR RSI returns to neutral
            if (close[i] <= sma_20_for_exit[i] or 
                rsi_12h_aligned[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals