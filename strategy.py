# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX (9,3) with volume confirmation and 1d ADX trend filter.
# TRIX is a triple-smoothed EMA momentum oscillator that filters noise and
# identifies trend changes. In trending markets (ADX > 25), we take TRIX
# crossovers with volume confirmation. In ranging markets (ADX < 20), we
# fade extreme TRIX readings (>0.1 or <-0.1) with volume confirmation.
# This adapts to both bull and bear markets by using ADX regime filter.
# Targets 20-40 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for ADX filter (once before loop)
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
    
    # Calculate ADX components
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
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate TRIX on 4h data (triple EMA of 9-period)
    close = prices['close'].values
    # First EMA
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value is zero
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        trix_val = trix[i]
        trix_prev = trix[i-1] if i > 0 else 0
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine market regime
            is_trending = adx_val > 25   # Trending market
            is_ranging = adx_val < 20    # Ranging market
            
            if is_trending:
                # Trending regime: TRIX crossover with volume
                if trix_val > 0 and trix_prev <= 0 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif trix_val < 0 and trix_prev >= 0 and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging regime: fade extreme TRIX readings
                if trix_val < -0.1 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif trix_val > 0.1 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on TRIX crossing below zero or extreme reading
                if trix_val < 0 or trix_val > 0.15:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on TRIX crossing above zero or extreme reading
                if trix_val > 0 or trix_val < -0.15:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_TRIX_ADX_Volume_Regime"
timeframe = "4h"
leverage = 1.0