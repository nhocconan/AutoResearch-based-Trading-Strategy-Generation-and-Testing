#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1-day EMA trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold reversal) AND price > EMA50(1d) AND volume > 1.5x 20-period average.
# Short when Williams %R crosses below -20 (overbought reversal) AND price < EMA50(1d) AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses back below -50 (long) or above -50 (short).
# Williams %R identifies reversals in overextended moves, effective in both bull and bear markets.
# EMA50 on daily timeframe filters trend direction. Volume confirms institutional participation.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag.

name = "4h_WilliamsR_1dEMA50_Volume"
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
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high - close_1d) / (highest_high - lowest_low))
    
    # EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80, price > EMA50, volume filter
            williams_r_cross_up = (williams_r_aligned[i] > -80) and (williams_r_aligned[i-1] <= -80)
            long_cond = williams_r_cross_up and (close[i] > ema_50_aligned[i]) and volume_filter[i]
            # Short conditions: Williams %R crosses below -20, price < EMA50, volume filter
            williams_r_cross_down = (williams_r_aligned[i] < -20) and (williams_r_aligned[i-1] >= -20)
            short_cond = williams_r_cross_down and (close[i] < ema_50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50
            williams_r_cross_down_exit = (williams_r_aligned[i] < -50) and (williams_r_aligned[i-1] >= -50)
            if williams_r_cross_down_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50
            williams_r_cross_up_exit = (williams_r_aligned[i] > -50) and (williams_r_aligned[i-1] <= -50)
            if williams_r_cross_up_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals