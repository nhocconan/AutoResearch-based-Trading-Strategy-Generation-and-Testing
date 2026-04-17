#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h breakout above/below weekly pivot levels (from prior week's daily data) 
# with volume confirmation and ADX filter for trend strength. 
# Uses weekly pivots to capture institutional levels, volume to confirm breakout validity, 
# and ADX to avoid ranging markets. Works in bull (breakouts continue) and bear (breakdowns continue) 
# by trading in direction of breakout with trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week's daily data (5 trading days)
    # Use shift(1) to avoid look-ahead: only use completed weeks
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(1).values
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
    # Weekly resistance and support levels
    weekly_r1 = 2 * weekly_pivot - low_5d
    weekly_s1 = 2 * weekly_pivot - high_5d
    weekly_r2 = weekly_pivot + (high_5d - low_5d)
    weekly_s2 = weekly_pivot - (high_5d - low_5d)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # ADX filter from 1d data (trend strength)
    # Calculate ADX(14) on daily data
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d_series - high_1d_series.shift(1)
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr_smooth = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr_smooth
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need weekly pivot, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or 
            np.isnan(weekly_r2_6h[i]) or 
            np.isnan(weekly_s2_6h[i]) or 
            np.isnan(adx_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_6h[i] > 20
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_6h[i]
        price_above_r2 = close[i] > weekly_r2_6h[i]
        price_below_s1 = close[i] < weekly_s1_6h[i]
        price_below_s2 = close[i] < weekly_s2_6h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R2 with volume and trending
            if (price_above_r2 and volume_filter and trending):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S2 with volume and trending
            elif (price_below_s2 and volume_filter and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly R1 OR ADX drops below 20 (trend weakening)
            if (close[i] < weekly_r1_6h[i]) or (adx_6h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly S1 OR ADX drops below 20 (trend weakening)
            if (close[i] > weekly_s1_6h[i]) or (adx_6h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_ADX_Volume_Breakout"
timeframe = "6h"
leverage = 1.0