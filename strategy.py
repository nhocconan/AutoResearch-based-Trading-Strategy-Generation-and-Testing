#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Donchian and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-day) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper band: highest high of last 20 days
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-day) on 1d data for trend strength filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR (14) on 4h data for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        adx_val = adx_1d_aligned[i]
        atr_4h_val = atr_4h[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(adx_val) or np.isnan(atr_4h_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with strong trend (ADX > 25)
            if close_val > upper_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with strong trend (ADX > 25)
            elif close_val < lower_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below lower Donchian band or ATR-based stop
            if close_val < lower_val or close_val < prices['high'].iloc[i] - 2.0 * atr_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian band or ATR-based stop
            if close_val > upper_val or close_val > prices['low'].iloc[i] + 2.0 * atr_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_DonchianBreakout_1dADXFilter_V1
# Uses 1-day Donchian channels (20) as breakout signals
# Enters long when 4h price breaks above 1d upper band with ADX > 25
# Enters short when 4h price breaks below 1d lower band with ADX > 25
# Uses 1d ADX as trend strength filter to avoid choppy markets
# Exits on opposite band touch or 2*ATR stoploss
# Designed for 4h timeframe with ~20-50 trades/year
name = "4h_DonchianBreakout_1dADXFilter_V1"
timeframe = "4h"
leverage = 1.0