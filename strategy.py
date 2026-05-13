#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND price > 12h HMA21 AND volume > 1.5x 20-period average volume.
# Short when price breaks below Donchian lower channel AND price < 12h HMA21 AND volume > 1.5x 20-period average volume.
# Exit when price crosses Donchian middle (20-period average) OR volume drops below average.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing
# institutional breakouts with trend and volume confirmation while avoiding false signals in low-volume environments.

name = "4h_DonchianBreakout_HMATrend_VolumeConfirm_v1"
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
    
    # Calculate 12h HMA21 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights / weights.sum(), mode='valid')
    
    if len(close_12h) < 21:
        hma_21_12h = np.full(len(close_12h), np.nan)
    else:
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        wma_2x_sub = 2 * wma_half[-len(wma_full):] - wma_full
        if len(wma_2x_sub) < sqrt_len:
            hma_21_12h = np.full(len(close_12h), np.nan)
        else:
            hma_21_12h = wma(wma_2x_sub, sqrt_len)
            # Pad with NaN to match original length
            hma_21_12h = np.concatenate([np.full(len(close_12h) - len(hma_21_12h), np.nan), hma_21_12h])
    
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    middle_channel = (highest_high + lowest_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    avg_volume = np.full(n, np.nan)
    for i in range(20 - 1, n):
        avg_volume[i] = np.mean(volume[i - 20 + 1:i + 1])
    volume_threshold = avg_volume * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(volume[i]) or 
            np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper channel AND price > 12h HMA21 AND volume > 1.5x avg volume
            if (close[i] > highest_high[i] and 
                close[i] > hma_21_12h_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower channel AND price < 12h HMA21 AND volume > 1.5x avg volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < hma_21_12h_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below middle channel OR volume drops below average
            if (close[i] < middle_channel[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above middle channel OR volume drops below average
            if (close[i] > middle_channel[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals