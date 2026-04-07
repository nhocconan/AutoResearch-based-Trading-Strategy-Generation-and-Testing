#!/usr/bin/env python3
"""
4h Bollinger Band Breakout with 1d Volume Confirmation and ADX Trend Filter
Long when price breaks above upper BB and volume > 1.5x average and ADX > 25
Short when price breaks below lower BB and volume > 1.5x average and ADX > 25
Exit when price crosses middle BB (mean reversion)
Designed to capture volatility breakouts in trending markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_breakout_1d_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    sma = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_std = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_bb = sma + bb_mult * bb_std
    lower_bb = sma - bb_mult * bb_std
    middle_bb = sma
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX (14) for trend strength
    adx_len = 14
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed values
    atr = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_len, min_periods=adx_len).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_len, min_periods=adx_len).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # 1d Volume Confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_1d_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(adx[i]) or np.isnan(vol_1d_ma_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below middle BB
            if close[i] < middle_bb[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle BB
            if close[i] > middle_bb[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * avg_volume[i]
            # 1d volume confirmation: current 1d volume > 1.5x average
            vol_1d_confirm = vol_1d_ma_aligned[i] > 0 and volume[i] > 1.5 * vol_1d_ma_aligned[i]
            # Trend filter: ADX > 25
            trend_filter = adx[i] > 25
            
            if close[i] > upper_bb[i] and vol_confirm and vol_1d_confirm and trend_filter:
                position = 1
                signals[i] = 0.30
            elif close[i] < lower_bb[i] and vol_confirm and vol_1d_confirm and trend_filter:
                position = -1
                signals[i] = -0.30
    
    return signals