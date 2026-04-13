#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary timeframe with 1w HTF filter
    # Long: price breaks above 1w Donchian(20) high + volume > 1.5x 20-period avg + ADX > 25 (strong trend)
    # Short: price breaks below 1w Donchian(20) low + volume > 1.5x 20-period avg + ADX > 25 (strong trend)
    # Exit: price returns to 1w Donchian middle (10-period average of high/low)
    # Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
    # Weekly Donchian breakouts with volume/ADX confirmation work in both bull/bear markets by capturing strong trends
    # Using 6h timeframe reduces trade frequency vs lower TFs while capturing major trends
    # Added: ADX filter to avoid choppy markets and reduce false breakouts
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for primary timeframe (weekly Donchian channels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.ones(len(df_1w))
    
    # Get 1d data for ADX calculation (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Donchian channels on 1w data (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate ADX on 1d data (14-period)
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        
        # +DM and -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        smoothed_plus_dm = pd.Series(plus_dm).rolling(window=window, min_periods=window).mean().values
        smoothed_minus_dm = pd.Series(minus_dm).rolling(window=window, min_periods=window).mean().values
        
        # Directional Indicators
        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=window, min_periods=window).mean().values
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, window=14)
    
    # Volume averages on 1d data (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: adx > 25 indicates strong trend (good for breakouts)
        is_strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume_1d[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close_1d[i] > donchian_high_aligned[i]
        breakout_down = close_1d[i] < donchian_low_aligned[i]
        
        # Entry conditions
        enter_long = is_strong_trend and breakout_up and volume_confirmed
        enter_short = is_strong_trend and breakout_down and volume_confirmed
        
        # Exit conditions: price returns to 1w Donchian middle
        exit_long = position == 1 and close_1d[i] <= donchian_mid_aligned[i]
        exit_short = position == -1 and close_1d[i] >= donchian_mid_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_donchian_breakout_volume_adx_v1"
timeframe = "6h"
leverage = 1.0