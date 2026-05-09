#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze with weekly pivot and volume confirmation.
# Uses Bollinger Band width to identify low volatility periods (squeeze).
# Breaks out in direction of weekly pivot (above pivot = long, below = short).
# Volume confirmation ensures breakout strength.
# Works in bull/bear by following weekly pivot as trend filter.
# Targets 15-25 trades/year to minimize fee drag.
name = "6h_BollingerSqueeze_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    bb_width = (upper - lower) / sma  # Normalized width
    
    # Weekly pivot points (standard)
    # Using previous week's data for current week's pivot
    prev_high = df_1w['high'].shift(1).values  # Previous week high
    prev_low = df_1w['low'].shift(1).values    # Previous week low
    prev_close = df_1w['close'].shift(1).values # Previous week close
    pivot = (prev_high + prev_low + prev_close) / 3
    
    # Weekly Bollinger Bands for trend context (optional filter)
    wb_period = 20
    wb_std = 2
    wb_sma = pd.Series(df_1w['close']).rolling(window=wb_period, min_periods=wb_period).mean().values
    wb_std_dev = pd.Series(df_1w['close']).rolling(window=wb_period, min_periods=wb_period).std().values
    wb_upper = wb_sma + (wb_std * wb_std_dev)
    wb_lower = wb_sma - (wb_std * wb_std_dev)
    
    # Align weekly data to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    wb_upper_aligned = align_htf_to_ltf(prices, df_1w, wb_upper)
    wb_lower_aligned = align_htf_to_ltf(prices, df_1w, wb_lower)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Bollinger squeeze condition: width < 20th percentile of last 50 values
        if i >= 50:
            width_history = bb_width[i-50:i]
            width_percentile = np.percentile(width_history[~np.isnan(width_history)], 20)
            squeeze = bb_width[i] < width_percentile
        else:
            squeeze = False
        
        if position == 0:
            # Enter long: squeeze + price > weekly pivot + volume confirmation
            if squeeze and price > pivot_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze + price < weekly pivot + volume confirmation
            elif squeeze and price < pivot_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly pivot or volatility expansion (width > 80th percentile)
            if i >= 50:
                width_history = bb_width[max(0, i-50):i+1]
                width_percentile_80 = np.percentile(width_history[~np.isnan(width_history)], 80)
                volatility_expansion = bb_width[i] > width_percentile_80
            else:
                volatility_expansion = False
                
            if price < pivot_aligned[i] or volatility_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly pivot or volatility expansion
            if i >= 50:
                width_history = bb_width[max(0, i-50):i+1]
                width_percentile_80 = np.percentile(width_history[~np.isnan(width_history)], 80)
                volatility_expansion = bb_width[i] > width_percentile_80
            else:
                volatility_expansion = False
                
            if price > pivot_aligned[i] or volatility_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals