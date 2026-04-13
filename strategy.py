#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h timeframe with 12h/1d HTF filters
    # Strategy: Williams Alligator + Elder Ray + Regime Filter
    # Long when: Jaw < Teeth < Lips (bullish alignment) AND Bull Power > 0 AND ADX > 25
    # Short when: Jaw > Teeth > Lips (bearish alignment) AND Bear Power < 0 AND ADX > 25
    # Exit: Alligator lines cross in opposite direction OR ADX < 20 (regime change)
    # Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
    # Alligator identifies trend, Elder Ray measures power, ADX filters for trending regimes
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Alligator and Elder Ray calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 12h data (13,8,5 SMAs with future shifts)
    # Jaw: 13-period SMA shifted 8 bars
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray on 12h data (13-period EMA)
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_12h = high_12h - ema_13_12h
    bear_power_12h = low_12h - ema_13_12h
    
    # ADX on 1d data (14-period)
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Directional Movement
        up_move = np.diff(high, prepend=high[0])
        down_move = -np.diff(low, prepend=low[0])
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / np.where(atr != 0, atr, np.nan)
        minus_di = 100 * minus_dm_smooth / np.where(atr != 0, atr, np.nan)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.nan)
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, window=14)
    
    # Align all indicators to 6h timeframe (primary)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Elder Ray power conditions
        bull_power_positive = bull_power_aligned[i] > 0
        bear_power_negative = bear_power_aligned[i] < 0
        
        # ADX regime filter: trending market
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20  # exit condition
        
        # Entry conditions
        enter_long = bullish_alignment and bull_power_positive and is_trending
        enter_short = bearish_alignment and bear_power_negative and is_trending
        
        # Exit conditions: Alligator cross in opposite direction OR ranging market
        exit_long = position == 1 and (not bullish_alignment or is_ranging)
        exit_short = position == -1 and (not bearish_alignment or is_ranging)
        
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

name = "6h_12h_1d_alligator_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0