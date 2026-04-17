#!/usr/bin/env python3
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
    
    # === 1d Bollinger Bands (20, 2) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate BBANDS
    sma_20 = np.zeros(len(close_1d))
    std_20 = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i >= 19:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
            std_20[i] = np.std(close_1d[i-19:i+1])
        else:
            sma_20[i] = np.nan
            std_20[i] = np.nan
    
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Calculate %B (position within bands)
    percent_b = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if std_20[i] > 0:
            percent_b[i] = (close_1d[i] - lower_band[i]) / (upper_band[i] - lower_band[i])
        else:
            percent_b[i] = 0.5
    
    # === 1d Volume Spike Detection ===
    vol_ma_20 = np.zeros(len(volume_1d))
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    
    vol_ratio = np.zeros(len(volume_1d))
    for i in range(len(volume_1d)):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume_1d[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    # === 1d RSI(14) for momentum confirmation ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(gain))
    avg_loss = np.zeros(len(loss))
    if len(gain) > 0:
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        for i in range(1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align to 4h timeframe
    percent_b_aligned = align_htf_to_ltf(prices, df_1d, percent_b)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(percent_b_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: significant spike
        vol_spike = vol_ratio_aligned[i] > 2.0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price at lower Bollinger Band (oversold) with volume spike and RSI not overbought
            if percent_b_aligned[i] <= 0.1 and vol_spike and rsi_aligned[i] < 70:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price at upper Bollinger Band (overbought) with volume spike and RSI not oversold
            elif percent_b_aligned[i] >= 0.9 and vol_spike and rsi_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to middle of bands
        elif position == 1:
            # Exit long: price returns to middle (50% level) or RSI overbought
            if percent_b_aligned[i] >= 0.5 or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle (50% level) or RSI oversold
            if percent_b_aligned[i] <= 0.5 or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Bollinger_Volume_Spike_RSI"
timeframe = "4h"
leverage = 1.0