#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation
    # Fade at R3/S3 (mean reversion in range), breakout continuation at R4/S4 (trend)
    # Volume spike confirms institutional interest. Works in bull/bear via dual regime logic.
    # Target: 15-30 trades/year to stay within 6h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # R4 = close + 1.5*(high - low), R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low), S4 = close - 1.5*(high - low)
    hl_range = high_1d - low_1d
    camarilla_r4 = close_1d + 1.5 * hl_range
    camarilla_r3 = close_1d + 1.1 * hl_range
    camarilla_s3 = close_1d - 1.1 * hl_range
    camarilla_s4 = close_1d - 1.5 * hl_range
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d index for volume check (4x per day in 6h TF)
        idx_1d = i // 4  # 6h bars per day = 4
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Mean reversion: fade at R3/S3 when price reaches extreme levels
        fade_long = (close[i] <= camarilla_s3_aligned[i]) and volume_confirmed
        fade_short = (close[i] >= camarilla_r3_aligned[i]) and volume_confirmed
        
        # Breakout continuation: break R4/S4 with volume
        breakout_long = (close[i] >= camarilla_r4_aligned[i]) and volume_confirmed
        breakout_short = (close[i] <= camarilla_s4_aligned[i]) and volume_confirmed
        
        # Stoploss: 1.5x ATR equivalent using Camarilla width
        camarilla_width = camarilla_r4_aligned[i] - camarilla_s4_aligned[i]
        stop_distance = camarilla_width * 0.02  # 2% of total Camarilla range
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals with priority: breakout > fade
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif fade_long and position != 1 and not (breakout_long or breakout_short):
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif fade_short and position != -1 and not (breakout_long or breakout_short):
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

name = "6h_1d_camarilla_pivot_breakout_fade_volume_v1"
timeframe = "6h"
leverage = 1.0