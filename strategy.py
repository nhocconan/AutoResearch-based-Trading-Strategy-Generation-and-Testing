#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w ADX trend filter
# - Long: price breaks above Donchian(20) high + volume surge + ADX > 25 (trending)
# - Short: price breaks below Donchian(20) low + volume surge + ADX > 25
# - Exit: opposite Donchian breakout or ADX < 20 (range)
# - Uses 1d volume surge (>1.5x 20-day average) for conviction
# - Designed to capture strong trends in both bull and bear markets
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_DonchianBreakout_1dVolume_1wADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    
    # Handle division by zero and NaN
    plus_di_1w = np.nan_to_num(plus_di_1w, nan=0.0)
    minus_di_1w = np.nan_to_num(minus_di_1w, nan=0.0)
    dx_1w = np.nan_to_num(dx_1w, nan=0.0)
    adx_1w = np.nan_to_num(adx_1w, nan=0.0)
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x scaled 1d average volume
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6.0)
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1w_aligned[i] > 25
        
        # Range filter: ADX < 20 indicates ranging market (exit condition)
        ranging = adx_1w_aligned[i] < 20
        
        if position == 0:
            # Look for long entry: breakout above Donchian high + volume + trending
            if close[i] > donchian_high[i] and volume_filter and trending:
                signals[i] = 0.25
                position = 1
            # Look for short entry: breakout below Donchian low + volume + trending
            elif close[i] < donchian_low[i] and volume_filter and trending:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on breakdown below Donchian low or ranging market
            if close[i] < donchian_low[i] or ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on breakout above Donchian high or ranging market
            if close[i] > donchian_high[i] or ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals