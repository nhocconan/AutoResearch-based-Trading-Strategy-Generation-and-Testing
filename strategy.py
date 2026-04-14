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
    
    # Load daily data (HTF) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate RSI on daily closes (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR (14-period) for volatility filter
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
    
    # Align RSI and ATR to 4h timeframe
    rsi_14_4h = align_htf_to_ltf(prices, df_1d, rsi_14)
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h price momentum (rate of change over 3 periods)
    roc_3 = np.full_like(close, np.nan)
    for i in range(3, len(close)):
        roc_3[i] = (close[i] - close[i-3]) / close[i-3] * 100
    
    # Volume spike detection (20-period average on 4h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_14_4h[i]) or 
            np.isnan(atr_14_4h[i]) or
            np.isnan(roc_3[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_14_4h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        if position == 0:
            # Long: RSI oversold (<30) + positive momentum + volume confirmation
            if (rsi_14_4h[i] < 30 and roc_3[i] > 0 and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought (>70) + negative momentum + volume confirmation
            elif (rsi_14_4h[i] > 70 and roc_3[i] < 0 and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: RSI returns to neutral (>50) or momentum turns negative
            if rsi_14_4h[i] > 50 or roc_3[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: RSI returns to neutral (<50) or momentum turns positive
            if rsi_14_4h[i] < 50 or roc_3[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_RSI_Momentum_Volume_Filter"
timeframe = "4h"
leverage = 1.0