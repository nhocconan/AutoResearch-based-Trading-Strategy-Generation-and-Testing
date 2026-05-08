#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) + 12h ADX (14) trend filter with volume confirmation
# Williams %R identifies overbought/oversold conditions; ADX filters for trending markets.
# Only trade when Williams %R is in extreme territory AND ADX > 25 (strong trend).
# Volume spike confirms momentum. Designed for low-frequency trades to minimize fee drag.
# Works in both bull and bear markets by adapting to trend direction via ADX.

name = "6h_WilliamsR_12hADX_Volume"
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
    
    # Get 12h data for Williams %R and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R (14) on 12h
    def calculate_williams_r(high, low, close, period):
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        for i in range(len(high)):
            if i >= period - 1:
                start_idx = i - period + 1
                highest_high[i] = np.max(high[start_idx:i+1])
                lowest_low[i] = np.min(low[start_idx:i+1])
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_12h = calculate_williams_r(high_12h, low_12h, close_12h, 14)
    
    # ADX (14) on 12h
    def calculate_adx(high, low, close, period):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - plus_dm[i-period+1] + plus_dm[i]
            minus_dm_sum = minus_dm_sum - minus_dm[i-period+1] + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align Williams %R and ADX to 6h timeframe
    wr_12h_aligned = align_htf_to_ltf(prices, df_12h, wr_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume spike (2.0x 20-period EMA) on 6h
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold (-80 or below), ADX > 25, volume spike
            if (wr_12h_aligned[i] <= -80 and 
                adx_12h_aligned[i] > 25 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (-20 or above), ADX > 25, volume spike
            elif (wr_12h_aligned[i] >= -20 and 
                  adx_12h_aligned[i] > 25 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 or ADX weakens
            if (wr_12h_aligned[i] > -50 or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 or ADX weakens
            if (wr_12h_aligned[i] < -50 or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals