#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (S1/S3) breakout with 1d volume spike and 1d ADX trend filter.
# Camarilla levels derived from prior day's range identify institutional support/resistance.
# Breakout above S3 or below S1 with volume > 2x 20-period average and ADX > 25
# indicates strong institutional participation. Designed for low trade frequency (~20-35/year)
# to minimize fee decay. Works in both bull and bear markets by following higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla and ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (S1, S3) from previous day
    # Range = (high - low) of previous day
    # S1 = close - 1.1 * (high - low) / 12
    # S3 = close - 1.1 * (high - low) / 4
    range_1d = high_1d - low_1d
    s1 = close_1d - 1.1 * range_1d / 12
    s3 = close_1d - 1.1 * range_1d / 4
    
    # Calculate ADX on 1d for trend strength
    # True Range
    tr0 = high_1d - low_1d
    tr1 = np.abs(high_1d - np.roll(close_1d, 1))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    tr[0] = tr0[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe (waits for 1d bar to close)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(s1_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        s1_val = s1_aligned[i]
        s3_val = s3_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        # Trend filter: ADX > 25
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long conditions: price breaks above S3 + volume spike + strong trend
            if price > s3_val and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 + volume spike + strong trend
            elif price < s1_val and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks back below S3 or trend weakens
                if price < s3_val or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks back above S1 or trend weakens
                if price > s1_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_S1S3_Breakout_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0