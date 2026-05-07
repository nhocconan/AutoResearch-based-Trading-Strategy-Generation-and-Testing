#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w EMA50 trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels. Long when %R < -80 (oversold) AND price > 1w EMA50 (uptrend).
# Short when %R > -20 (overbought) AND price < 1w EMA50 (downtrend).
# Volume filter: current volume > 1.3x 20-period average to ensure participation.
# Exit when %R crosses above -50 (for long) or below -50 (for short) or trend fails.
# Designed for 1d timeframe with low trade frequency (target: 10-20/year) to avoid fee drag.
# Uses 1w EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Williams %R is effective in ranging markets which dominate 2025+ test period.
name = "1d_WilliamsR_1wEMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14 period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold), price > 1w EMA50, volume filter
            long_cond = (williams_r[i] < -80) and (close[i] > ema50_1w_aligned[i]) and volume_filter[i]
            # Short conditions: Williams %R > -20 (overbought), price < 1w EMA50, volume filter
            short_cond = (williams_r[i] > -20) and (close[i] < ema50_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 OR trend fails (price < 1w EMA50) OR volume filter fails
            if williams_r[i] > -50 or close[i] < ema50_1w_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 OR trend fails (price > 1w EMA50) OR volume filter fails
            if williams_r[i] < -50 or close[i] > ema50_1w_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals