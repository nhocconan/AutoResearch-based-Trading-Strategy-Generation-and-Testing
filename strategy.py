#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX trend filter
    # Bull Power = High - EMA13 (buying pressure)
    # Bear Power = Low - EMA13 (selling pressure)
    # Long: Bull Power > 0 and Bear Power < 0 (bullish momentum) + 1d ADX > 25 (strong trend)
    # Short: Bull Power < 0 and Bear Power > 0 (bearish momentum) + 1d ADX > 25 (strong trend)
    # Exit: Opposite Elder Ray signal or ADX < 20 (trend weakening)
    # Uses 1d ADX for regime filter to avoid whipsaws in ranging markets
    # Elder Ray captures momentum extremes, ADX ensures we only trade strong trends
    # Works in bull (buy strength) and bear (sell weakness) with trend confirmation
    # Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fees
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for ADX trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray components
    bull_power = high_6h - ema_13_6h  # Buying pressure
    bear_power = low_6h - ema_13_6h   # Selling pressure
    
    # Calculate 1d ADX (14-period) for trend strength
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(atr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    di_minus = 100 * dm_minus_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # start from 30 to have enough data for ADX
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0
        bearish_momentum = bull_power[i] < 0 and bear_power[i] > 0
        
        # Trend filter from 1d ADX
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        # Entry conditions
        long_entry = bullish_momentum and strong_trend and position != 1
        short_entry = bearish_momentum and strong_trend and position != -1
        
        # Exit conditions
        exit_long = position == 1 and (not bullish_momentum or weak_trend)
        exit_short = position == -1 and (not bearish_momentum or weak_trend)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
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

name = "6h_1d_elder_ray_adx_filter_v1"
timeframe = "6h"
leverage = 1.0