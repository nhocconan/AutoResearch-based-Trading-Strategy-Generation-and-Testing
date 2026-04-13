#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h primary with 12h HTF - Camarilla pivot breakout + volume spike + chop regime filter
    # Works in bull/bear by trading strong intraday moves with volume confirmation while avoiding ranging markets
    # Target: 100-200 trades over 4 years (25-50/year) for optimal fee drag balance
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for HTF Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 4h data for volume confirmation and chop filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We'll use R3 and S3 as breakout levels
    prev_high_12h = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low_12h = np.concatenate([[np.nan], low_12h[:-1]])
    prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
    
    camarilla_r3 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h)
    camarilla_s3 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h)
    
    # Calculate 4h volume spike (2x 20-period average)
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Choppiness Index regime filter (CHOP > 61.8 = ranging, avoid)
    def calculate_chop(high, low, close, window=14):
        atr_sum = pd.Series(np.abs(high - low)).rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop.values
    
    chop_4h = calculate_chop(high_4h, low_4h, close_4h, window=14)
    
    # Align all HTF/LTF indicators to 4h primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(chop_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirmed = volume_4h[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Chop regime filter: avoid ranging markets (CHOP > 61.8)
        regime_filter = chop_4h_aligned[i] <= 61.8
        
        # Breakout conditions
        breakout_up = close_4h[i] > camarilla_r3_aligned[i]
        breakout_down = close_4h[i] < camarilla_s3_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and regime_filter
        enter_short = breakout_down and volume_confirmed and regime_filter
        
        # Exit conditions: price returns to previous 12h close (pivot point)
        exit_long = position == 1 and close_4h[i] <= prev_close_12h[i]  # using previous 12h close as pivot
        exit_short = position == -1 and close_4h[i] >= prev_close_12h[i]
        
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

name = "4h_12h_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0