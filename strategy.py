#!/usr/bin/env python3
"""
4h_PriceChannel_Breakout_Volume_Regime
Strategy: 4h price channel breakout (Donchian) with volume confirmation and Choppiness regime filter.
Long: Price breaks above Donchian(20) high + volume > 1.5x average + Choppiness > 61.8 (range)
Short: Price breaks below Donchian(20) low + volume > 1.5x average + Choppiness > 61.8 (range)
Exit: Opposite breakout or trend change (EMA34 crossover)
Designed for 4h timeframe: ~20-40 trades/year per symbol (80-160 total over 4 years).
Uses Choppiness to avoid whipsaws in strong trends, focus on range-breakouts that tend to revert.
Works in bull/bear via regime filter that adapts to market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # EMA34 for trend filter
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Choppiness indicator
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed +DM, -DM, TR (14-period)
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX (14-period)
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: 100 * log10(sum(ATR)/ (n * (highest high - lowest low))) / log10(n)
    # Using 14-period as standard
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (14 * (highest_high - lowest_low))) / np.log10(14)
    
    # Align all 4h data to 4h timeframe (no alignment needed as we're already in 4h)
    # Align daily Choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align other indicators to 4h (though they're already aligned, keeping for consistency)
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition (EMA34)
        uptrend = close_4h[i] > ema_34_aligned[i]
        downtrend = close_4h[i] < ema_34_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_4h[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Choppiness regime filter: only trade in ranging markets (Choppiness > 61.8)
        ranging = chop_aligned[i] > 61.8
        
        # Breakout conditions
        breakout_up = high_4h[i] > high_20_aligned[i]
        breakout_down = low_4h[i] < low_20_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + ranging market
            if breakout_up and vol_confirm and ranging:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + ranging market
            elif breakout_down and vol_confirm and ranging:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: downward breakout or trend change to downtrend
            if breakout_down or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: upward breakout or trend change to uptrend
            if breakout_up or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceChannel_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0