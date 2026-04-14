#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA and MACD
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate MACD(12,26,9) on 1d
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False).values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False).values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).values
    macd_hist = macd_line - signal_line
    
    # Align MACD components to 6h timeframe
    macd_hist_6h = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) on 1d for volatility filter
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
    
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(macd_hist_6h[i]) or 
            np.isnan(ema50_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: MACD histogram crosses above zero with volume and above EMA50
            if (macd_hist_6h[i] > 0 and 
                macd_hist_6h[i-1] <= 0 and
                volume_ratio > vol_threshold and
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = position_size
            # Short: MACD histogram crosses below zero with volume and below EMA50
            elif (macd_hist_6h[i] < 0 and 
                  macd_hist_6h[i-1] >= 0 and
                  volume_ratio > vol_threshold and
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: MACD histogram crosses below zero
            if macd_hist_6h[i] < 0 and macd_hist_6h[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: MACD histogram crosses above zero
            if macd_hist_6h[i] > 0 and macd_hist_6h[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_MACD_EMA50_Volume_Filter"
timeframe = "6h"
leverage = 1.0