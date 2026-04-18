#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams %R (14-period)
    period_wr = 14
    highest_high = pd.Series(high_1d).rolling(window=period_wr, min_periods=period_wr).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period_wr, min_periods=period_wr).min().values
    wr = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # ADX (14-period)
    period_adx = 14
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period_adx, min_periods=period_adx).mean().values
    
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period_adx, min_periods=period_adx).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period_adx, min_periods=period_adx).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period_adx, min_periods=period_adx).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 6h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for ADX and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend strength condition: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x daily average
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Williams %R conditions
        oversold = wr_aligned[i] < -80
        overbought = wr_aligned[i] > -20
        
        if position == 0:
            # Long: strong trend + volume + oversold
            if strong_trend and vol_confirm and oversold:
                signals[i] = 0.25
                position = 1
            # Short: strong trend + volume + overbought
            elif strong_trend and vol_confirm and overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakens or overbought condition
            if not strong_trend or overbought:
                signals[i] = 0.0  # exit long
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakens or oversold condition
            if not strong_trend or oversold:
                signals[i] = 0.0  # exit short
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_ADX_Volume"
timeframe = "6h"
leverage = 1.0