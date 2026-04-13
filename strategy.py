#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h timeframe with 1d HTF
    # Strategy: Camarilla pivot levels from 1d data
    # Long: price breaks above R4 with volume confirmation in trending regime (ADX > 25)
    # Short: price breaks below S4 with volume confirmation in trending regime (ADX > 25)
    # Exit: price returns to daily pivot point (PP)
    # Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
    # Camarilla pivots work well in both trending and ranging markets when combined with ADX filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla pivots and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Camarilla pivot levels for 1d data
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3
    r4 = pp + (high_1d - low_1d) * 1.1 / 2
    s4 = pp - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate ADX on 1d data (14-period)
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        tr_smooth = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        dm_plus_smooth = pd.Series(dm_plus).rolling(window=window, min_periods=window).mean().values
        dm_minus_smooth = pd.Series(dm_minus).rolling(window=window, min_periods=window).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
        di_minus = 100 * dm_minus_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
        adx = pd.Series(dx).rolling(window=window, min_periods=window).mean().values
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, window=14)
    
    # Volume average on 1d data (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trending regime filter: ADX > 25 indicates strong trend
        is_trending = adx_aligned[i] > 25
        
        # Volume confirmation: current 1d volume > 1.3x 20-day average
        volume_confirmed = volume_1d[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions at Camarilla R4/S4 levels
        breakout_up = close_1d[i] > r4_aligned[i]
        breakout_down = close_1d[i] < s4_aligned[i]
        
        # Entry conditions
        enter_long = is_trending and breakout_up and volume_confirmed
        enter_short = is_trending and breakout_down and volume_confirmed
        
        # Exit conditions: price returns to daily pivot point (PP)
        exit_long = position == 1 and close_1d[i] <= pp_aligned[i]
        exit_short = position == -1 and close_1d[i] >= pp_aligned[i]
        
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

name = "6h_1d_camarilla_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0