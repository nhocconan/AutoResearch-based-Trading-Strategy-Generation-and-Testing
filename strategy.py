#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_ADXFilter
Hypothesis: Breakout above/below 4h Donchian(20) with volume spike and ADX>25 confirms strong momentum. 
Exit when price crosses back below/above the 4h 10-period EMA or ADX weakens (<20). Designed for low trade frequency 
to avoid fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ADX(14) trend strength filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - low[:-1]), np.absolute(low[1:] - high[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA(10) for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20, 20, 14*2)  # Need warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(adx[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(ema_10[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = volume_spike[i]
        adx_val = adx[i]
        ema_1d = ema_34_1d_aligned[i]
        ema_10_val = ema_10[i]
        
        if position == 0:
            # Long: price > Donchian upper with volume spike, strong trend (ADX>25), and above daily EMA34
            if price > upper and vol_spike and adx_val > 25 and price > ema_1d:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower with volume spike, strong trend (ADX>25), and below daily EMA34
            elif price < lower and vol_spike and adx_val > 25 and price < ema_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < EMA10 OR ADX weakens (<20)
            if price < ema_10_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > EMA10 OR ADX weakens (<20)
            if price > ema_10_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_20_Breakout_Volume_Trend_ADXFilter"
timeframe = "4h"
leverage = 1.0