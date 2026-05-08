#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Bollinger Bands squeeze breakout with volume confirmation and ATR stop.
# Bollinger Band squeeze (low volatility) often precedes strong breakout moves.
# Long when price breaks above upper BB with volume spike; short when breaks below lower BB.
# Uses 1-day trend filter (EMA50) to align with higher timeframe direction.
# Designed for low trade frequency (15-25/year) to minimize fee drag and capture explosive moves.
# Works in bull markets (breaks to upside) and bear markets (breaks to downside).

name = "4h_BB_Squeeze_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (std_20 * bb_std)
    lower_bb = sma_20 - (std_20 * bb_std)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width for squeeze detection
    
    # Bollinger Band squeeze: width below 20th percentile of last 50 days
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Daily trend filter: EMA50 slope
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope = ema_50_1d[1:] > ema_50_1d[:-1]  # Rising EMA50
    ema50_slope = np.concatenate([[False], ema50_slope])  # Align with daily index
    
    # Align indicators to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope.astype(float))
    
    # Volume confirmation: current volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(bb_squeeze_aligned[i]) or np.isnan(ema50_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: BB squeeze breakout above upper band with volume
            if (bb_squeeze_aligned[i] > 0.5 and  # In squeeze condition
                close[i] > upper_bb_aligned[i] and  # Break above upper BB
                ema50_slope_aligned[i] > 0.5 and    # Daily uptrend
                vol_confirm[i]):                    # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short setup: BB squeeze breakout below lower band with volume
            elif (bb_squeeze_aligned[i] > 0.5 and   # In squeeze condition
                  close[i] < lower_bb_aligned[i] and  # Break below lower BB
                  ema50_slope_aligned[i] <= 0.5 and   # Daily downtrend
                  vol_confirm[i]):                    # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle of BB or trend changes
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if close[i] < middle_bb or ema50_slope_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle of BB or trend changes
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if close[i] > middle_bb or ema50_slope_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals