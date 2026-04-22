#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought).
# 1d ADX > 25 filters for trending markets to avoid false signals in ranging conditions.
# In trending markets (ADX > 25): buy when Williams %R crosses above -80 from below, sell when crosses below -20 from above.
# Uses volume spike (1.5x 20-period average) to confirm momentum.
# Designed to capture momentum swings in both bull and bear markets by trading with the trend.
# Targets 15-35 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Williams %R and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate ADX components (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6x Williams %R crossovers for entry signals
    williams_r_prev = np.roll(williams_r_aligned, 1)
    williams_r_prev[0] = williams_r_aligned[0]
    
    # Bullish crossover: Williams %R crosses above -80 from below
    bullish_cross = (williams_r_aligned > -80) & (williams_r_prev <= -80)
    # Bearish crossover: Williams %R crosses below -20 from above
    bearish_cross = (williams_r_aligned < -20) & (williams_r_prev >= -20)
    
    # Calculate 6-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_6[i]
        bullish = bullish_cross[i]
        bearish = bearish_cross[i]
        
        # Volume filter: current volume > 1.5 * 6-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx_val > 25
        
        if position == 0:
            # Enter long on bullish Williams %R crossover in trending market with volume spike
            if is_trending and bullish and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish Williams %R crossover in trending market with volume spike
            elif is_trending and bearish and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on bearish crossover or loss of trend
                if bearish or not is_trending:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on bullish crossover or loss of trend
                if bullish or not is_trending:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_ADX_Volume_Trend"
timeframe = "6h"
leverage = 1.0