# US Equities: MACD + Volume + Trend Filter Strategy
# Hypothesis: Combines MACD momentum with volume confirmation and trend filter for robust entries in both bull and bear markets
# Uses 6h timeframe with 1d trend filter to capture multi-timeframe confluence
# Designed to avoid overtrading with strict entry conditions targeting 12-37 trades/year

name = "6h_MACD_Volume_Trend_Filter"
timeframe = "6h"
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
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (ema_200_1d[i-1] * 199 + close_1d[i]) / 200
    
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate MACD (12,26,9)
    ema_12 = np.full_like(close, np.nan)
    ema_26 = np.full_like(close, np.nan)
    
    if len(close) >= 12:
        ema_12[11] = np.mean(close[0:12])
        for i in range(12, len(close)):
            ema_12[i] = (close[i] * 2/13) + (ema_12[i-1] * 11/13)
    
    if len(close) >= 26:
        ema_26[25] = np.mean(close[0:26])
        for i in range(26, len(close)):
            ema_26[i] = (close[i] * 2/27) + (ema_26[i-1] * 25/27)
    
    macd_line = np.full_like(close, np.nan)
    valid_macd = (~np.isnan(ema_12)) & (~np.isnan(ema_26))
    macd_line[valid_macd] = ema_12[valid_macd] - ema_26[valid_macd]
    
    signal_line = np.full_like(close, np.nan)
    valid_signal = ~np.isnan(macd_line)
    if np.sum(valid_signal) >= 9:
        signal_line[8] = np.mean(macd_line[0:9])
        for i in range(9, len(close)):
            if valid_signal[i]:
                signal_line[i] = (macd_line[i] * 2/10) + (signal_line[i-1] * 8/10)
    
    macd_histogram = np.full_like(close, np.nan)
    valid_hist = (~np.isnan(macd_line)) & (~np.isnan(signal_line))
    macd_histogram[valid_hist] = macd_line[valid_hist] - signal_line[valid_hist]
    
    # Volume ratio: current / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20, 200)  # Ensure MACD, volume MA, and trend filter are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(macd_histogram[i]) or np.isnan(signal_line[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: MACD histogram crosses above zero AND uptrend (price > EMA200) AND volume spike
            if (macd_histogram[i] > 0 and macd_histogram[i-1] <= 0 and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: MACD histogram crosses below zero AND downtrend (price < EMA200) AND volume spike
            elif (macd_histogram[i] < 0 and macd_histogram[i-1] >= 0 and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: MACD histogram crosses below zero OR trend reversal (price < EMA200)
            if (macd_histogram[i] < 0 and macd_histogram[i-1] >= 0) or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: MACD histogram crosses above zero OR trend reversal (price > EMA200)
            if (macd_histogram[i] > 0 and macd_histogram[i-1] <= 0) or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals