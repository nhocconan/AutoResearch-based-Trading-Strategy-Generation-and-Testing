#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily volume moving average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate daily ADX (14-period) for trend strength
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate daily 60-period high/low for Donchian channel
    high_60 = pd.Series(high_1d).rolling(window=60, min_periods=60).max().values
    low_60 = pd.Series(low_1d).rolling(window=60, min_periods=60).min().values
    high_60_aligned = align_htf_to_ltf(prices, df_1d, high_60)
    low_60_aligned = align_htf_to_ltf(prices, df_1d, low_60)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(high_60_aligned[i]) or 
            np.isnan(low_60_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price breaks above 60-day high, strong trend (ADX > 25), volume confirmation
            if (price > high_60_aligned[i] and 
                adx_1d_aligned[i] > 25 and 
                vol > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 60-day low, strong trend (ADX > 25), volume confirmation
            elif (price < low_60_aligned[i] and 
                  adx_1d_aligned[i] > 25 and 
                  vol > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below 60-day low or trend weakens (ADX < 20)
            if price < low_60_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above 60-day high or trend weakens (ADX < 20)
            if price > high_60_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian60_ADX25_VolumeFilter"
timeframe = "6h"
leverage = 1.0