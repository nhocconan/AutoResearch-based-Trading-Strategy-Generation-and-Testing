#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly EMA(20) for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev * 2) / 4
    range_ = high_prev - low_prev
    
    # Focus on R3/S3 levels for mean reversion entries
    r3 = pivot + range_ * 1.25
    s3 = pivot - range_ * 1.25
    
    # Align levels to daily timeframe (already aligned via index matching)
    ema34_aligned = ema34_1d_aligned
    ema20w_aligned = ema20_1w_aligned
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, pivots, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema20w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_aligned[i]
        weekly_trend = ema20w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Mean reversion at S3: price touches level and reverses in uptrend
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i] and 
                close[i] > ema_trend and close[i] > weekly_trend and vol_spike_val):
                signals[i] = size
                position = 1
            # Mean reversion at R3: price touches level and reverses in downtrend
            elif (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i] and 
                  close[i] < ema_trend and close[i] < weekly_trend and vol_spike_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R3 (mean reversion target) or trend reverses
            if high[i] >= r3_aligned[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches S3 (mean reversion target) or trend reverses
            if low[i] <= s3_aligned[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3S3_MeanReversion_1dEMA34_1wEMA20_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0