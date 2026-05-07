#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with weekly trend filter and volume spike.
# Uses 1d price action for entries, 1w EMA for trend filter, and volume confirmation.
# Designed to capture breakouts in trending markets while avoiding range-bound whipsaws.
# Target: 10-25 trades/year per symbol to minimize fee drag.
name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for price action and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 8:
        return np.zeros(n)
    
    # 1w trend filter: 8-period EMA on close
    ema_8_1w = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Previous day's OHLC for Camarilla calculation (1d)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels for current day (based on previous day)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = prev_close + prev_range * 1.1 / 4
    camarilla_s3 = prev_close - prev_range * 1.1 / 4
    
    # 1d volume spike: current volume > 2.0x 20-period SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = np.where(vol_sma_20 > 0, volume / vol_sma_20, 1.0) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for 1d
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_8_1w_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1w EMA8
        uptrend = close[i] > ema_8_1w_aligned[i]
        downtrend = close[i] < ema_8_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price > R3 with volume spike in uptrend
            long_condition = (close[i] > camarilla_r3[i]) and vol_spike[i] and uptrend
            # Short breakdown: price < S3 with volume spike in downtrend
            short_condition = (close[i] < camarilla_s3[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below R3 or trend turns down
            if (close[i] < camarilla_r3[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above S3 or trend turns up
            if (close[i] > camarilla_s3[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals