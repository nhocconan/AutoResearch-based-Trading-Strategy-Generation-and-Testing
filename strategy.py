#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ATR-based Donchian breakout with volume confirmation and ADX trend filter.
# Uses 1-day timeframe for ATR calculation and 12h for entry/exit logic.
# Designed to capture strong trending moves with low frequency to minimize fee drag.
# Entry: Long when price breaks above Donchian upper band with volume spike and ADX > 25.
# Short when price breaks below Donchian lower band with volume spike and ADX > 25.
# Exit: Opposite Donchian band touch or ADX drops below 20 (trend weakening).
# Position size: 0.25 (25% of capital) to balance risk and return.
# Target: 15-30 trades/year to avoid overtrading and fee drag.

name = "12h_ATR_Donchian_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR-based Donchian channels (20-period) on 12h timeframe
    # Using ATR to dynamically adjust channel width
    atr_factor = 1.0
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Alternative: ATR-adjusted bands (more adaptive)
    # upper_band = pd.Series(close).rolling(window=20, min_periods=20).mean().values + (atr_14 * atr_factor)
    # lower_band = pd.Series(close).rolling(window=20, min_periods=20).mean().values - (atr_14 * atr_factor)
    
    # Align ATR to 12h timeframe (needed for ATR-adjusted bands if used)
    # For standard Donchian, we don't need to align ATR since we're using price-based bands
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # ADX calculation (14-period) on 12h timeframe
    # +DM, -DM, TR
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * (plus_dm_smooth / atr_12h)
    minus_di = 100 * (minus_dm_smooth / atr_12h)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Handle division by zero and NaN values
    plus_di = np.where(atr_12h == 0, 0, plus_di)
    minus_di = np.where(atr_12h == 0, 0, minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = np.where(np.isnan(adx), 0, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with volume spike and strong trend (ADX > 25)
            if (close[i] > upper_band[i] and 
                volume_spike[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike and strong trend (ADX > 25)
            elif (close[i] < lower_band[i] and 
                  volume_spike[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower band or trend weakens (ADX < 20)
            if (close[i] < lower_band[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper band or trend weakens (ADX < 20)
            if (close[i] > upper_band[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals