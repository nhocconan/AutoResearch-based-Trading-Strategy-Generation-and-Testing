#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for weekly high/low (range)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly range and position within range
    weekly_range = high_1w - low_1w
    weekly_position = (close - low_1w) / weekly_range  # 0 = at weekly low, 1 = at weekly high
    weekly_position = np.where(weekly_range == 0, 0.5, weekly_position)  # avoid div by zero
    
    # Align weekly data to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    weekly_pos_aligned = align_htf_to_ltf(prices, df_1w, weekly_position)
    
    # Get daily data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR(14) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR to daily timeframe (no change but for consistency)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate volume ratio: current volume / 20-day average volume
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_ratio = volume_1d / np.where(vol_ma_20_aligned == 0, np.nan, vol_ma_20_aligned)
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)  # default to average when no data
    
    # Align volume ratio to daily timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_pos_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions:
        # 1. Price near weekly extremes (oversold/overbought)
        # 2. High volume (confirmation of interest)
        # 3. Mean reversion expectation
        
        weekly_pos = weekly_pos_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        if position == 0:
            # Long: near weekly low with high volume (capitulation bounce)
            if weekly_pos < 0.2 and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: near weekly high with high volume (exhaustion)
            elif weekly_pos > 0.8 and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price moves back to middle of weekly range or volatility drops
            if weekly_pos > 0.5 or vol_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price moves back to middle of weekly range or volatility drops
            if weekly_pos < 0.5 or vol_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyExtremes_VolumeMeanReversion"
timeframe = "1d"
leverage = 1.0