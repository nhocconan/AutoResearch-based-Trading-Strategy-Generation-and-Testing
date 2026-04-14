#!/usr/bin/env python3
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
    
    # Load daily data for calculations (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 10-period EMA for trend on daily
    if len(close_1d) < 10:
        return np.zeros(n)
    ema_10 = np.full(len(close_1d), np.nan)
    close_series = pd.Series(close_1d)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).values
    
    # Calculate 5-period EMA for signal on daily
    if len(close_1d) < 5:
        return np.zeros(n)
    ema_5 = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).values
    
    # Calculate daily ATR for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily indicators to 6h timeframe
    ema_10_6h = align_htf_to_ltf(prices, df_1d, ema_10)
    ema_5_6h = align_htf_to_ltf(prices, df_1d, ema_5)
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 60-period volume moving average on 6h for volume filter
    vol_ma_60 = np.full_like(volume, np.nan)
    if len(volume) >= 60:
        vol_series = pd.Series(volume)
        vol_ma_60 = vol_series.rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_10_6h[i]) or 
            np.isnan(ema_5_6h[i]) or
            np.isnan(atr_1d_6h[i]) or
            np.isnan(vol_ma_60[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d_6h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 60-period average
        if vol_ma_60[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_60[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        if position == 0:
            # Long: EMA 5 crosses above EMA 10 with volume confirmation
            if (ema_5_6h[i] > ema_10_6h[i] and ema_5_6h[i-1] <= ema_10_6h[i-1] and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: EMA 5 crosses below EMA 10 with volume confirmation
            elif (ema_5_6h[i] < ema_10_6h[i] and ema_5_6h[i-1] >= ema_10_6h[i-1] and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: EMA 5 crosses below EMA 10
            if ema_5_6h[i] < ema_10_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: EMA 5 crosses above EMA 10
            if ema_5_6h[i] > ema_10_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_EMA_Crossover_Volume_Filter"
timeframe = "6h"
leverage = 1.0