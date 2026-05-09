#!/usr/bin/env python3
"""
4h_EMA_Crossover_RSI_Filter_MACD_Confirmation
Hypothesis: A medium-term trend strategy using EMA crossovers (9/21) on 4h timeframe, filtered by RSI(14) for momentum and MACD(12,26,9) for confirmation. 
Designed to work in both bull and bear markets by only taking trades in the direction of the higher timeframe trend (1d EMA50). 
Low trade frequency is ensured by requiring multiple confirmations, reducing whipsaws and fee impact. 
Target: 20-50 trades per year.
"""

name = "4h_EMA_Crossover_RSI_Filter_MACD_Confirmation"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA9 and EMA21 on 4h for crossover
    ema9 = np.full_like(close, np.nan)
    ema21 = np.full_like(close, np.nan)
    
    if len(close) >= 9:
        ema9[8] = np.mean(close[0:9])
        for i in range(9, len(close)):
            ema9[i] = (ema9[i-1] * 8 + close[i]) / 9
    
    if len(close) >= 21:
        ema21[20] = np.mean(close[0:21])
        for i in range(21, len(close)):
            ema21[i] = (ema21[i-1] * 20 + close[i]) / 21
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD(12,26,9)
    ema12 = np.full_like(close, np.nan)
    ema26 = np.full_like(close, np.nan)
    
    if len(close) >= 12:
        ema12[11] = np.mean(close[0:12])
        for i in range(12, len(close)):
            ema12[i] = (ema12[i-1] * 11 + close[i]) / 12
    
    if len(close) >= 26:
        ema26[25] = np.mean(close[0:26])
        for i in range(26, len(close)):
            ema26[i] = (ema26[i-1] * 25 + close[i]) / 26
    
    macd_line = np.subtract(ema12, ema26)
    signal_line = np.full_like(close, np.nan)
    
    if len(close) >= 35:  # 26+9
        signal_line[34] = np.mean(macd_line[26:35])  # first 9 values
        for i in range(35, len(close)):
            signal_line[i] = (signal_line[i-1] * 8 + macd_line[i]) / 9
    
    macd_histogram = macd_line - signal_line
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma20[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    
    volume_filter = np.full_like(volume, np.nan)
    valid_vol = ~np.isnan(vol_ma20) & (vol_ma20 > 0)
    volume_filter[valid_vol] = volume[valid_vol] / vol_ma20[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(35, 21, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(rsi[i]) or np.isnan(macd_histogram[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: EMA9 > EMA21, RSI > 50, MACD histogram > 0, uptrend (price > EMA50_1d), volume confirmation
            if (ema9[i] > ema21[i] and 
                rsi[i] > 50 and 
                macd_histogram[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: EMA9 < EMA21, RSI < 50, MACD histogram < 0, downtrend (price < EMA50_1d), volume confirmation
            elif (ema9[i] < ema21[i] and 
                  rsi[i] < 50 and 
                  macd_histogram[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: EMA9 < EMA21 OR RSI < 40 OR MACD histogram < 0
                if (ema9[i] < ema21[i] or 
                    rsi[i] < 40 or 
                    macd_histogram[i] < 0):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: EMA9 > EMA21 OR RSI > 60 OR MACD histogram > 0
                if (ema9[i] > ema21[i] or 
                    rsi[i] > 60 or 
                    macd_histogram[i] > 0):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals