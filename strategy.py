#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot R3/S3 Breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla pivots provide intraday support/resistance levels based on prior day's range.
# Breakouts above R3 or below S3 indicate strong momentum with institutional participation.
# 1w EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation (>2.0x 20-period EMA) validates the breakout strength.
# Designed for low trade frequency (15-25/year) to minimize fee drag on 1d timeframe.
# Works in both bull and bear markets by aligning with the weekly trend.

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start from 2 to have prior day's data for pivots
        # Skip if any value is NaN
        if np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels for today using yesterday's OHLC
        # Camarilla formula: 
        # R4 = close + ((high-low) * 1.5/2)
        # R3 = close + ((high-low) * 1.25/2)
        # R2 = close + ((high-low) * 1.1/2)
        # R1 = close + ((high-low) * 1.05/2)
        # PP = (high + low + close) / 3
        # S1 = close - ((high-low) * 1.05/2)
        # S2 = close - ((high-low) * 1.1/2)
        # S3 = close - ((high-low) * 1.25/2)
        # S4 = close - ((high-low) * 1.5/2)
        
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        prev_range = prev_high - prev_low
        
        # Avoid division by zero or invalid calculations
        if prev_range <= 0 or np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_r3 = prev_close + (prev_range * 1.25 / 2)
        camarilla_s3 = prev_close - (prev_range * 1.25 / 2)
        
        # Volume confirmation: 20-period EMA on 1d volume
        if i >= 20:
            vol_series = pd.Series(volume[:i+1])
            vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
            volume_spike = volume[i] > (2.0 * vol_ema_20)
        else:
            volume_spike = False
        
        # Breakout signals with 1w trend filter
        # Long: Price breaks above Camarilla R3 + above 1w EMA50 + volume spike
        # Short: Price breaks below Camarilla S3 + below 1w EMA50 + volume spike
        if position == 0:
            if (close[i] > camarilla_r3 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < camarilla_s3 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price drops below Camarilla R3 OR below 1w EMA50
            if close[i] < camarilla_r3 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price rises above Camarilla S3 OR above 1w EMA50
            if close[i] > camarilla_s3 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals