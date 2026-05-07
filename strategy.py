#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) with 1d trend filter and volume spike.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 with volume spike.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 with volume spike.
# Uses 1d EMA50 trend filter to align with higher timeframe trend and avoid counter-trend trades.
# Volume spike filter ensures momentum confirmation. Designed for fewer trades (target: 15-25/year) to reduce fee drag.
# Works in both bull and bear markets by following the 1d trend direction.
name = "6h_WilliamsR_1dEMA50_Volume"
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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend filter: 50-period EMA on close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) - using high, low, close
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values  # Convert to numpy array
    
    # 6h volume average for spike detection
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long condition: Williams %R oversold (< -80), in uptrend with volume spike
            long_condition = (williams_r[i] < -80) and uptrend and vol_spike[i]
            # Short condition: Williams %R overbought (> -20), in downtrend with volume spike
            short_condition = (williams_r[i] > -20) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R rises above -50 (moving out of oversold) or trend turns down
            if (williams_r[i] > -50) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R falls below -50 (moving out of overbought) or trend turns up
            if (williams_r[i] < -50) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals