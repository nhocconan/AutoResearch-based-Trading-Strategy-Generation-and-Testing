#!/usr/bin/env python3
"""
12h Multi-Timeframe Strategy: Combines 12h Donchian breakout with 1d EMA trend filter, 
volume confirmation, and 1d Choppiness Index regime filter. 
Long when price breaks above 12h Donchian upper band with volume > 1.5x average, 
price above 1d EMA50, and CHOP > 61.8 (trending regime). 
Short when price breaks below 12h Donchian lower band with volume > 1.5x average, 
price below 1d EMA50, and CHOP > 61.8. 
Exit on opposite Donchian breakout or when CHOP < 38.2 (range regime).
Designed for 12h to work in trending markets with ~20-30 trades per year.
"""

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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align 12h Donchian levels to lower timeframe
    donch_high_ltf = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_ltf = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_ltf = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # Get 1d data for EMA and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_ltf = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(TR) / (HHV - LLV)) / log10(period)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    hh = df_1d['high'].rolling(window=14, min_periods=14).max()
    ll = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_values = chop.values
    chop_ltf = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume confirmation: 20-period volume MA on current timeframe
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_ltf[i]) or np.isnan(donch_low_ltf[i]) or 
            np.isnan(ema_50_ltf[i]) or np.isnan(chop_ltf[i]) or 
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: Donchian breakout up + volume + EMA filter + trending regime
            if (price > donch_high_ltf[i] and vol > 1.5 * vol_ma and 
                price > ema_50_ltf[i] and chop_ltf[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume + EMA filter + trending regime
            elif (price < donch_low_ltf[i] and vol > 1.5 * vol_ma and 
                  price < ema_50_ltf[i] and chop_ltf[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: opposite breakout OR range regime (CHOP < 38.2)
            if price < donch_low_ltf[i] or chop_ltf[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite breakout OR range regime (CHOP < 38.2)
            if price > donch_high_ltf[i] or chop_ltf[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_EMA_CHOP_Volume_Breakout"
timeframe = "12h"
leverage = 1.0