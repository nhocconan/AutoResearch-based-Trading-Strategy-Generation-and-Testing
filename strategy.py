#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 12h Camarilla pivot levels with 1d volume spike confirmation
    # Works in both bull and bear: Camarilla levels provide mean-reversion at R3/S3 and breakout signals at R4/S4
    # Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee drag.
    # Target: 12-30 trades/year to stay within 6h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for Camarilla pivot calculation (primary HTF for structure)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 12h Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    # Use previous 12h bar's OHLC for current bar's levels (no look-ahead)
    prev_high_12h = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low_12h = np.concatenate([[np.nan], low_12h[:-1]])
    prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
    
    camarilla_pp = (prev_high_12h + prev_low_12h + prev_close_12h) / 3
    camarilla_range = prev_high_12h - prev_low_12h
    
    camarilla_r4 = camarilla_pp + (camarilla_range * 1.1 / 2)
    camarilla_r3 = camarilla_pp + (camarilla_range * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (camarilla_range * 1.1 / 4)
    camarilla_s4 = camarilla_pp - (camarilla_range * 1.1 / 2)
    
    # Get 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average (spike detection)
        idx_1d = i // 4  # 4 six-hour bars per day
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_spike = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Mean reversion at R3/S3: price touches extreme level and reverses
        mean_revert_long = (close[i] <= camarilla_s3_aligned[i]) and volume_spike
        mean_revert_short = (close[i] >= camarilla_r3_aligned[i]) and volume_spike
        
        # Breakout continuation at R4/S4: price breaks extreme level with volume
        breakout_long = (close[i] > camarilla_r4_aligned[i]) and volume_spike
        breakout_short = (close[i] < camarilla_s4_aligned[i]) and volume_spike
        
        # Stoploss: based on Camarilla width (R3-S3 range)
        camarilla_width = camarilla_r3_aligned[i] - camarilla_s3_aligned[i]
        stop_distance = camarilla_width * 0.15  # 15% of R3-S3 range
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if (mean_revert_long or breakout_long) and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif (mean_revert_short or breakout_short) and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_12h_1d_camarilla_pivot_volume_spike_v1"
timeframe = "6h"
leverage = 1.0