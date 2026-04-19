#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ADX trend filter
# Uses daily high/low to calculate Donchian channels, avoiding look-ahead bias
# Long when price breaks above upper band with volume spike and ADX > 25
# Short when price breaks below lower band with volume spike and ADX > 25
# Exit when price crosses the middle band (average of upper/lower) or ATR-based stop
# Designed for low frequency (~15-25 trades/year) to minimize fee drag
name = "12h_Donchian20_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-day high and low
    high_20d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (waits for daily close)
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # Middle band for exit
    mid_band = (high_20d_aligned + low_20d_aligned) / 2
    
    # ADX(14) for trend strength
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20d_aligned[i]) or np.isnan(low_20d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with volume and trend
            if (close[i] > high_20d_aligned[i] and 
                volume_spike[i] and 
                adx[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short: break below lower band with volume and trend
            elif (close[i] < low_20d_aligned[i] and 
                  volume_spike[i] and 
                  adx[i] > 25):
                signals[i] = -0.30
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses middle band or ADX weakens
            if (close[i] < mid_band[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:
            # Short: exit if price crosses middle band or ADX weakens
            if (close[i] > mid_band[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals