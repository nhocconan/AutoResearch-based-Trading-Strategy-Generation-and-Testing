#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with weekly pivot structure and volume confirmation
# In low volatility regimes (BB width < 20th percentile), price is primed for explosive moves
# Breakout direction determined by weekly Camarilla pivot (above/below weekly pivot = bullish/bearish bias)
# Volume confirmation filters false breakouts
# Works in bull markets via upside breakouts and bear markets via downside breakouts during squeeze periods
# Target: 12-37 trades/year on 6h timeframe to minimize fee drag

name = "6h_BB_Squeeze_WeeklyCamarilla_Pivot_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Bollinger Bands
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized BB width
    
    # Align BB width to 6h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 20th percentile of BB width for squeeze detection (using expanding window)
    bb_width_percentile = np.full_like(bb_width_aligned, np.nan)
    for i in range(len(bb_width_aligned)):
        if i >= 20:  # Need minimum history for percentile
            hist = bb_width_aligned[max(0, i-100):i+1]  # Last 100 values or available
            if len(hist) >= 20:
                bb_width_percentile[i] = np.percentile(hist[~np.isnan(hist)], 20)
    
    # 1w HTF data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly Camarilla pivot calculation (based on prior week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    camarilla_h4 = np.full_like(close_1w, np.nan)  # Resistance level 4
    camarilla_l4 = np.full_like(close_1w, np.nan)  # Support level 4
    camarilla_pivot = np.full_like(close_1w, np.nan)  # Pivot point
    
    for i in range(1, len(close_1w)):  # Start from 1 to use prior week data
        if not (np.isnan(high_1w[i-1]) or np.isnan(low_1w[i-1]) or np.isnan(close_1w[i-1])):
            camarilla_pivot[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3
            range_1w = high_1w[i-1] - low_1w[i-1]
            camarilla_h4[i] = camarilla_pivot[i] + (range_1w * 1.1 / 2)
            camarilla_l4[i] = camarilla_pivot[i] - (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume confirmation: current volume > 1.5 * 50-period average volume
    volume_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (volume_ma_50 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need sufficient history for volume MA and BB
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width_aligned[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width below 20th percentile (low volatility)
        squeeze = bb_width_aligned[i] < bb_width_percentile[i]
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h4_aligned[i]
        breakout_down = close[i] < camarilla_l4_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: squeeze breakout above H4 with volume
            if squeeze and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout below L4 with volume
            elif squeeze and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price returns to pivot or squeeze breaks down
            if close[i] <= camarilla_pivot_aligned[i] or not squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to pivot or squeeze breaks up
            if close[i] >= camarilla_pivot_aligned[i] or not squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals