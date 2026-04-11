#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v2
# Strategy: 12h Camarilla pivot level touch with 1d volume confirmation and ADX trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price tends to revert to Camarilla pivot levels (H3/L3) in ranging markets.
# Volume spike confirms participation at these key levels. ADX < 20 ensures ranging regime.
# Designed for low trade frequency to minimize fee drag. Works in both bull and bear markets
# via mean reversion at statistically significant support/resistance levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # H3 = Close + 1.1*(High - Low)/2
    # L3 = Close - 1.1*(High - Low)/2
    # H4 = Close + 1.5*(High - Low)/2
    # L4 = Close - 1.5*(High - Low)/2
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # 1d ADX (14-period) for ranging regime filter
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    
    # Calculate ADX
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Ranging regime: ADX < 20
        ranging = adx_14_aligned[i] < 20
        
        # Price proximity to Camarilla levels (within 0.1% tolerance)
        tolerance = 0.001
        near_h3 = abs(close[i] - camarilla_h3_aligned[i]) / camarilla_h3_aligned[i] < tolerance
        near_l3 = abs(close[i] - camarilla_l3_aligned[i]) / camarilla_l3_aligned[i] < tolerance
        near_h4 = abs(close[i] - camarilla_h4_aligned[i]) / camarilla_h4_aligned[i] < tolerance
        near_l4 = abs(close[i] - camarilla_l4_aligned[i]) / camarilla_l4_aligned[i] < tolerance
        
        # Entry conditions
        # Long: Near L3/L4 support AND volume confirmation AND ranging market
        if ((near_l3 or near_l4) and vol_confirm and ranging and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: Near H3/H4 resistance AND volume confirmation AND ranging market
        elif ((near_h3 or near_h4) and vol_confirm and ranging and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Price moves to opposite side of pivot level
        elif position == 1 and (close[i] > camarilla_h3_aligned[i] or close[i] < camarilla_l3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_h3_aligned[i] or close[i] < camarilla_l3_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals