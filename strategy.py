#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (above -20 = overbought, below -80 = oversold)
# In trending markets (ADX > 25), we fade extremes: short when Williams %R > -20, long when < -80
# In ranging markets (ADX <= 25), we mean-revert at extreme levels with volume confirmation
# Volume filter requires current volume > 1.5 * 20-period average
# Designed for 12h timeframe to capture multi-day swings while minimizing trade frequency
# Works in both bull and bear markets by adapting to trend strength via ADX

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for ADX and Williams %R calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM for ADX
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM (14-period)
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align ADX and Williams %R to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 12h price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        wr = williams_r_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine market regime
            is_trending = adx_val > 25  # Strong trend
            is_ranging = adx_val <= 25  # Weak trend/ranging
            
            if is_trending:
                # Trending regime: fade extreme Williams %R readings
                if wr < -80 and vol_spike:  # Oversold - go long
                    signals[i] = 0.25
                    position = 1
                elif wr > -20 and vol_spike:  # Overbought - go short
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging regime: mean reversion at extreme Williams %R levels
                if wr < -80 and vol_spike:  # Deep oversold - go long
                    signals[i] = 0.25
                    position = 1
                elif wr > -20 and vol_spike:  # Deep overbought - go short
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R returns to neutral territory (> -50) or overbought
                if wr > -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R returns to neutral territory (< -50) or oversold
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_ADX_Volume_Filter"
timeframe = "12h"
leverage = 1.0