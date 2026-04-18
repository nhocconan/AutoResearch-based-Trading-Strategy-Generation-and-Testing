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
    
    # Get daily data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channels
    upper_donchian = np.full_like(high_1d, np.nan)
    lower_donchian = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        upper_donchian[i] = np.max(high_1d[i-20:i])
        lower_donchian[i] = np.min(low_1d[i-20:i])
    
    # Calculate 14-day ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - np.concatenate([[high_1d[0]], high_1d[:-1]]))
    tr3 = np.abs(low_1d[1:] - np.concatenate([[low_1d[0]], low_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = np.full_like(tr, np.nan)
    if len(tr) >= 15:
        atr_1d[14] = np.mean(tr[1:15])
        for i in range(15, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    if len(close_1w) >= 50:
        ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all 1d data to 4h timeframe
    upper_4h = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_4h = align_htf_to_ltf(prices, df_1d, lower_donchian)
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(ema_1w_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA(50)
        bullish_bias = close[i] > ema_1w_4h[i]
        bearish_bias = close[i] < ema_1w_4h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume in bullish bias
            if close[i] > upper_4h[i] and vol_confirm and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume in bearish bias
            elif close[i] < lower_4h[i] and vol_confirm and bearish_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR trend turns bearish
            if close[i] < lower_4h[i] or not bullish_bias:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR trend turns bullish
            if close[i] > upper_4h[i] or bullish_bias:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Filter_v2"
timeframe = "4h"
leverage = 1.0