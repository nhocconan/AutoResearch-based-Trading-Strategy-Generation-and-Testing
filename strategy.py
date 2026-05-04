#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extremes + 1d ADX Trend Filter + Volume Confirmation
# Williams %R identifies overbought/oversold conditions. 1d ADX > 25 ensures trending market.
# Volume confirmation ensures conviction. Designed for 12-37 trades/year on 6h to minimize fee drag.
# Works in bull markets via long when %R < -80 (oversold) in uptrend and in bear markets via short when %R > -20 (overbought) in downtrend.

name = "6h_WilliamsR_Extremes_1dADX_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX for trend filter
    def calculate_adx(high, low, close, period=14):
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
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Williams %R
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        for i in range(len(high)):
            start_idx = max(0, i - period + 1)
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) AND 1d uptrend (ADX > 25) AND volume spike
            if (williams_r[i] < -80 and   # Oversold
                adx_1d_aligned[i] > 25 and  # Strong trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) AND 1d downtrend (ADX > 25) AND volume spike
            elif (williams_r[i] > -20 and   # Overbought
                  adx_1d_aligned[i] > 25 and  # Strong trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR ADX falls below 20
            if williams_r[i] > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR ADX falls below 20
            if williams_r[i] < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals