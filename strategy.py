#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA200 for trend filter
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # 4h ATR for volatility filter
    tr4 = np.maximum(high_4h - low_4h,
                     np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                np.abs(low_4h - np.roll(close_4h, 1))))
    tr4[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr4).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        
        if position == 0:
            # Long: price above 4h EMA200 with volume confirmation and sufficient volatility
            if (price > ema_200_4h_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                atr_4h_aligned[i] > 0):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA200 with volume confirmation and sufficient volatility
            elif (price < ema_200_4h_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  atr_4h_aligned[i] > 0):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 4h EMA200 or volatility drops significantly
            if price < ema_200_4h_aligned[i] or vol < 0.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h EMA200 or volatility drops significantly
            if price > ema_200_4h_aligned[i] or vol < 0.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA200_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0