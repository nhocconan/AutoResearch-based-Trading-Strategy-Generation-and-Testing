#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 with volume spike.
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 with volume spike.
# Uses 1d EMA34 trend filter to align with higher timeframe trend and avoid counter-trend trades.
# Volume spike filter ensures momentum confirmation. Designed for fewer trades (target: 20-30/year) to reduce fee drag.
# Works in both bull and bear markets by following the 1d trend direction.
name = "4h_Camarilla_R3S3_1dEMA34_Volume"
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
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels (based on previous day's range)
    # Calculate using previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # 4h volume average for spike detection
    vol_ema_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long condition: break above Camarilla R3, in uptrend with volume spike
            long_condition = (close[i] > camarilla_r3[i]) and uptrend and vol_spike[i]
            # Short condition: break below Camarilla S3, in downtrend with volume spike
            short_condition = (close[i] < camarilla_s3[i]) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Camarilla S3 or trend turns down
            if (close[i] < camarilla_s3[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Camarilla R3 or trend turns up
            if (close[i] > camarilla_r3[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals