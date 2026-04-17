#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Bollinger Band breakout + volume confirmation + 12h ADX trend filter.
Long when price breaks above 12h BB upper band with volume confirmation and 12h ADX > 25 (strong trend).
Short when price breaks below 12h BB lower band with volume confirmation and 12h ADX > 25 (strong trend).
Exit when price returns to 12h BB middle band (20-period SMA) or reverses with volume.
Uses 12h timeframe for structure (reduces noise) and 4h for entry timing and volume confirmation.
Designed to capture strong trending moves with institutional volume while avoiding false breakouts in ranging markets.
Bollinger Bands adapt to volatility, making them effective in both bull and bear markets when combined with ADX trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Bollinger Bands and ADX calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Bollinger Bands (20, 2)
    close_12h_series = pd.Series(close_12h)
    sma20_12h = close_12h_series.rolling(window=20, min_periods=20).mean().values
    std20_12h = close_12h_series.rolling(window=20, min_periods=20).std().values
    upper_bb_12h = sma20_12h + 2 * std20_12h
    lower_bb_12h = sma20_12h - 2 * std20_12h
    middle_bb_12h = sma20_12h  # 20-period SMA
    
    # Calculate 12h ADX (14) for trend filter
    # ADX requires +DI, -DI, and DX calculation
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    close_12h_series = pd.Series(close_12h)
    
    # Calculate True Range
    tr1 = high_12h_series - low_12h_series
    tr2 = abs(high_12h_series - close_12h_series.shift(1))
    tr3 = abs(low_12h_series - close_12h_series.shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Calculate +DM and -DM
    up_move = high_12h_series.diff()
    down_move = low_12h_series.shift(1) - low_12h_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di_14 = 100 * (plus_dm_14 / tr_14)
    minus_di_14 = 100 * (minus_dm_14 / tr_14)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    upper_bb_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_bb_12h)
    lower_bb_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_bb_12h)
    middle_bb_12h_aligned = align_htf_to_ltf(prices, df_12h, middle_bb_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for Bollinger Bands and ADX calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_12h_aligned[i]) or 
            np.isnan(lower_bb_12h_aligned[i]) or 
            np.isnan(middle_bb_12h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 12h BB upper band with volume and strong trend (ADX > 25)
            if (close[i] > upper_bb_12h_aligned[i] and 
                volume_confirmed and 
                adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h BB lower band with volume and strong trend (ADX > 25)
            elif (close[i] < lower_bb_12h_aligned[i] and 
                  volume_confirmed and 
                  adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below middle band OR breaks below lower band with volume (reversal)
            if (close[i] <= middle_bb_12h_aligned[i] or 
                (close[i] < lower_bb_12h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above middle band OR breaks above upper band with volume (reversal)
            if (close[i] >= middle_bb_12h_aligned[i] or 
                (close[i] > upper_bb_12h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hBB_Breakout_Volume_ADX25_Trend"
timeframe = "4h"
leverage = 1.0