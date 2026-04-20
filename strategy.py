#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for price action, volume and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume average for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA for trend (34-period EMA)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly high/low for range detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    ema_high_1w = pd.Series(high_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_low_1w = pd.Series(low_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_high_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_high_1w)
    ema_low_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_low_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_high_1w_aligned[i]) or 
            np.isnan(ema_low_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        # Weekly range width for normalization
        weekly_range = ema_high_1w_aligned[i] - ema_low_1w_aligned[i]
        if weekly_range <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Position of price within weekly EMA range (0 = low, 1 = high)
        price_position = (price - ema_low_1w_aligned[i]) / weekly_range
        
        if position == 0:
            # Long: price in upper 40% of weekly range with volume expansion and volatility
            if (price_position > 0.6 and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                atr_1d_aligned[i] > 0.5 * atr_1d_aligned[max(0, i-20)]):  # volatility expanding
                signals[i] = 0.25
                position = 1
            # Short: price in lower 40% of weekly range with volume expansion and volatility
            elif (price_position < 0.4 and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  atr_1d_aligned[i] > 0.5 * atr_1d_aligned[max(0, i-20)]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 50% level or volume drops
            if price_position < 0.5 or vol < 0.8 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 50% level or volume drops
            if price_position > 0.5 or vol < 0.8 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyRangePosition_VolumeExpansion"
timeframe = "6h"
leverage = 1.0