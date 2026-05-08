#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Bands breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper BB AND price > EMA50(1d) AND volume > 1.8x 20-period average.
# Short when price breaks below lower BB AND price < EMA50(1d) AND volume > 1.8x 20-period average.
# Exit when price crosses back below middle band (long) or above middle band (short).
# Bollinger Bands measure volatility and price extremes; EMA50 filters trend direction; volume confirms.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag.

name = "4h_BollingerBands_1dEMA50_Volume"
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
    
    # 4h volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    # 1d data for Bollinger Bands and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Bollinger Bands parameters
    bb_period = 20
    bb_std = 2.0
    
    # Calculate Bollinger Bands from 1d close
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    
    # EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper BB, price > EMA50, volume filter
            long_cond = (close[i] > bb_upper_aligned[i]) and (close[i] > ema_50_aligned[i]) and volume_filter[i]
            # Short conditions: break below lower BB, price < EMA50, volume filter
            short_cond = (close[i] < bb_lower_aligned[i]) and (close[i] < ema_50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below middle BB
            if close[i] < bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above middle BB
            if close[i] > bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals