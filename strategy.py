#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Ensure enough data for calculations
        return np.zeros(n)
    
    # Load daily data for trend and volatility (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA21 for trend filter
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Calculate daily ATR for volatility filter (14-period)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily volume average for confirmation (10-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema21_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price above daily EMA21 with volume confirmation and sufficient volatility
            if (price > ema21_1d_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                atr_1d_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA21 with volume confirmation and sufficient volatility
            elif (price < ema21_1d_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  atr_1d_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA21 or volume drops significantly
            if price < ema21_1d_aligned[i] or vol < 0.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA21 or volume drops significantly
            if price > ema21_1d_aligned[i] or vol < 0.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA21_VolumeFilter"
timeframe = "1d"
leverage = 1.0