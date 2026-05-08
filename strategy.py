#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Bollinger Band squeeze and 1w EMA trend filter with volume confirmation.
# Bollinger Band squeeze (BB width < 20th percentile) indicates low volatility and potential breakout.
# Long when price breaks above upper BB with volume surge and above 1w EMA.
# Short when price breaks below lower BB with volume surge and below 1w EMA.
# Designed for low trade frequency (20-40/year) to avoid fee drag. Works in both trending and ranging markets by capturing volatility breakouts.

name = "4h_BB_Squeeze_1wEMA_Volume"
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
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20, 2)
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + (2 * std_20)
    lower_bb = ma_20 - (2 * std_20)
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(20, len(bb_width)):
        window = bb_width[i-20:i]
        bb_width_percentile[i] = (bb_width[i] <= window).sum() / len(window) * 100
    
    # Squeeze condition: BB width below 20th percentile
    squeeze = bb_width_percentile < 20
    
    # Calculate 1w EMA (using 5-day EMA as proxy)
    ema_1w = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_1w)
    
    # Volume confirmation: 4h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper BB + BB squeeze + volume surge + above 1w EMA
            if close[i] > upper_bb_aligned[i] and squeeze_aligned[i] and vol_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB + BB squeeze + volume surge + below 1w EMA
            elif close[i] < lower_bb_aligned[i] and squeeze_aligned[i] and vol_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower BB
            if close[i] < lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper BB
            if close[i] > upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals