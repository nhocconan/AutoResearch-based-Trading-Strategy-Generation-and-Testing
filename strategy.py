#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla breakout with 1d volume and chop regime filter
    # Enter long when price breaks above R3 with volume > 1.5x 20-bar avg and CHOP < 61.8 (trending)
    # Enter short when price breaks below S3 with volume > 1.5x 20-bar avg and CHOP < 61.8
    # Exit when price crosses the 1d midpoint (close)
    # Uses 1d HTF for Camarilla levels and chop regime, 12h for entry timing
    # Camarilla R3/S3 provide strong institutional levels
    # Volume confirmation ensures breakout participation
    # Chop filter avoids whipsaws in ranging markets
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for Camarilla pivot calculation and chop regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    cam_high_low = high_1d - low_1d
    camarilla_r3 = close_1d + (cam_high_low * 1.1 / 4)
    camarilla_s3 = close_1d - (cam_high_low * 1.1 / 4)
    camarilla_mid = close_1d  # midpoint is the 1d close
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(1), n) / (max(high,n) - min(low,n))) / log10(n)
    # Where n=14 period, CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first period
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_1d = chop_raw  # already in 0-100 range
    
    # Align 1d Chopiness Index to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure indicators are ready
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop_1d_aligned[i] < 61.8
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]  # break above R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # break below S3
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = breakout_up and volume_confirmed[i] and trending_regime and position != 1
        short_entry = breakout_down and volume_confirmed[i] and trending_regime and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_mid_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_mid_aligned[i])
        
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

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0