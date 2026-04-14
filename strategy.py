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
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly EMA(21) for trend
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate median volume for volume spike filter
    vol_median = np.nanmedian(volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned daily data
        atr_1d_i = align_htf_to_ltf(prices, df_1d, atr_1d)[i]
        ema_21_1w_i = align_htf_to_ltf(prices, df_1w, ema_21_1w)[i]
        
        if np.isnan(ema_21_1w_i) or np.isnan(atr_1d_i):
            continue
        
        # Volatility filter: only trade when daily ATR is above median (avoid choppy markets)
        atr_median = np.nanmedian(atr_1d)
        vol_filter = atr_1d_i > 0.8 * atr_median
        
        # Volume spike filter
        volume_spike = volume[i] > 1.5 * vol_median
        
        # Long conditions:
        # 1. Price above weekly EMA21 (uptrend)
        # 2. Volatility filter
        # 3. Volume spike
        if position == 0 and vol_filter and volume_spike:
            if close[i] > ema_21_1w_i:
                position = 1
                signals[i] = position_size
            # Short conditions:
            # 1. Price below weekly EMA21 (downtrend)
            elif close[i] < ema_21_1w_i:
                position = -1
                signals[i] = -position_size
        
        # Exit conditions: price crosses back across weekly EMA21
        elif position == 1:
            if close[i] < ema_21_1w_i:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            if close[i] > ema_21_1w_i:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA21_Volume_Volatility_Filter"
timeframe = "6h"
leverage = 1.0