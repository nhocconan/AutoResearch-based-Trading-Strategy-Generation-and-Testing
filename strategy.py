#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate daily ATR-based range filter (avoid low volatility periods)
    atr_ma = pd.Series(atr14_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    volatility_filter = atr14_1d > (0.5 * atr_ma)  # Only trade when volatility is above half its MA
    
    # Calculate daily Camarilla levels from previous day
    high_1d_shift = df_1d['high'].shift(1).values
    low_1d_shift = df_1d['low'].shift(1).values
    close_1d_shift = df_1d['close'].shift(1).values
    
    # Calculate pivot and Camarilla levels using previous day's data
    pivot = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_ = high_1d_shift - low_1d_shift
    r3 = close_1d_shift + (range_ * 1.1 / 4)
    s3 = close_1d_shift - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(atr_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        vol_filter = volatility_filter[i]
        
        if position == 0:
            # Enter long: price breaks above S3 + uptrend + volume spike + volatility filter
            if (close[i] > s3_val and 
                close[i] > ema34_1d_val and 
                vol_spike and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below R3 + downtrend + volume spike + volatility filter
            elif (close[i] < r3_val and 
                  close[i] < ema34_1d_val and 
                  vol_spike and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR trend turns down
            if (close[i] < s3_val or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR trend turns up
            if (close[i] > r3_val or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals