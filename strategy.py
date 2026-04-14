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
    
    # Load weekly data once for trend and structure
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high over past 20 weeks
    donchian_upper = np.full(len(high_1w), np.nan)
    for i in range(20, len(high_1w)):
        donchian_upper[i] = np.max(high_1w[i-20:i])
    
    # Lower band: lowest low over past 20 weeks
    donchian_lower = np.full(len(low_1w), np.nan)
    for i in range(20, len(low_1w)):
        donchian_lower[i] = np.min(low_1w[i-20:i])
    
    # Calculate weekly ATR for volatility filter
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close[1:])  # Using close for simplicity
    tr3 = np.abs(low_1w[1:] - close[:-1])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full(len(tr_1w), np.nan)
    for i in range(14, len(tr_1w)):
        atr_1w[i] = np.nanmean(tr_1w[i-13:i+1])
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Create arrays for alignment
    donchian_upper_arr = donchian_upper
    donchian_lower_arr = donchian_lower
    atr_1w_arr = atr_1w
    ema_20_1w_arr = ema_20_1w
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Get aligned weekly data
        donchian_upper_i = align_htf_to_ltf(prices, df_1w, donchian_upper_arr)[i]
        donchian_lower_i = align_htf_to_ltf(prices, df_1w, donchian_lower_arr)[i]
        atr_1w_i = align_htf_to_ltf(prices, df_1w, atr_1w_arr)[i]
        ema_20_1w_i = align_htf_to_ltf(prices, df_1w, ema_20_1w_arr)[i]
        
        if np.isnan(donchian_upper_i) or np.isnan(donchian_lower_i) or np.isnan(atr_1w_i) or np.isnan(ema_20_1w_i):
            continue
        
        # Volatility filter: only trade when ATR is above median (avoid chop)
        if atr_1w_i < np.nanmedian(atr_1w):
            continue
        
        # Long entry: price breaks above weekly Donchian upper + above weekly EMA20
        if position == 0:
            if close[i] > donchian_upper_i and close[i] > ema_20_1w_i:
                position = 1
                signals[i] = position_size
        # Long exit: price breaks below weekly Donchian lower
        elif position == 1:
            if close[i] < donchian_lower_i:
                position = 0
                signals[i] = 0.0
        
        # Short entry: price breaks below weekly Donchian lower + below weekly EMA20
        if position == 0:
            if close[i] < donchian_lower_i and close[i] < ema_20_1w_i:
                position = -1
                signals[i] = -position_size
        # Short exit: price breaks above weekly Donchian upper
        elif position == -1:
            if close[i] > donchian_upper_i:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchianBreakout_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0