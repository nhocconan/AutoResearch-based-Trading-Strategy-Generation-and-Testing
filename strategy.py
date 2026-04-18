#!/usr/bin/env python3
"""
4h_4F_Trend_Filter_Breakout
Hypothesis: Combines 4-hour Donchian breakout with 1-day EMA34 trend filter, volume confirmation, and ADX strength filter. 
Designed for low trade frequency (target: 20-50/year) with strong performance in both bull and bear markets by requiring 
multiple confluence factors before entry, reducing false signals and whipsaws.
"""

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
    
    # Calculate 1-day EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1-day EMA34
    ema34_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema34_1d[i] = np.mean(close_1d[0:35])
        else:
            k = 2 / (34 + 1)
            ema34_1d[i] = close_1d[i] * k + ema34_1d[i-1] * (1 - k)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1-day ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    tr_period = 14
    atr_1d = np.full(len(close_1d), np.nan)
    plus_dm_smooth = np.full(len(close_1d), np.nan)
    minus_dm_smooth = np.full(len(close_1d), np.nan)
    
    for i in range(tr_period, len(close_1d)):
        if i == tr_period:
            atr_1d[i] = np.nanmean(tr[1:i+1])
            plus_dm_smooth[i] = np.nanmean(plus_dm[1:i+1])
            minus_dm_smooth[i] = np.nanmean(minus_dm[1:i+1])
        else:
            atr_1d[i] = atr_1d[i-1] - (atr_1d[i-1] / tr_period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / tr_period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / tr_period) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di_1d = np.full(len(close_1d), np.nan)
    minus_di_1d = np.full(len(close_1d), np.nan)
    dx_1d = np.full(len(close_1d), np.nan)
    
    for i in range(tr_period, len(close_1d)):
        if atr_1d[i] > 0:
            plus_di_1d[i] = 100 * (plus_dm_smooth[i] / atr_1d[i])
            minus_di_1d[i] = 100 * (minus_dm_smooth[i] / atr_1d[i])
            if (plus_di_1d[i] + minus_di_1d[i]) > 0:
                dx_1d[i] = 100 * np.abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i])
    
    # Calculate ADX
    adx_1d = np.full(len(close_1d), np.nan)
    for i in range(2*tr_period, len(close_1d)):
        valid_dx = dx_1d[tr_period:i+1]
        if len(valid_dx) >= tr_period:
            adx_1d[i] = np.nanmean(valid_dx[-tr_period:])
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4-hour Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 2*14)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike, uptrend, and strong ADX
            if (close[i] > donchian_high[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i] and adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike, downtrend, and strong ADX
            elif (close[i] < donchian_low[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i] and adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Donchian low or trend turns down
            if (close[i] < donchian_low[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or trend turns up
            if (close[i] > donchian_high[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4F_Trend_Filter_Breakout"
timeframe = "4h"
leverage = 1.0