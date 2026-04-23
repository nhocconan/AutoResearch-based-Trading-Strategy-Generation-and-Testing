#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above R3 and 1d ADX > 25 with volume > 1.5x average.
Short when price breaks below S3 and 1d ADX > 25 with volume > 1.5x average.
Exit on opposite Camarilla level break or ADX < 20 (trend weakening).
Camarilla levels provide precise support/resistance from prior day's range.
1d ADX > 25 filters for strong trending days to avoid false breakouts in chop.
Volume confirmation ensures breakout legitimacy.
Designed for 12h timeframe targeting 50-150 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts in direction of strong trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (np.sum(plus_dm[i-period+1:i+1]) / atr[i]) * 100
                minus_di[i] = (np.sum(minus_dm[i-period+1:i+1]) / atr[i]) * 100
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pivot = (high + low + close) / 3
        range_val = high - low
        
        # Camarilla levels
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        r4 = pivot + (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1 / 2)
        
        return r3, s3, r4, s4
    
    # Calculate Camarilla levels for each 1d bar (using prior bar's data)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        r3, s3, _, _ = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R3 AND 1d ADX > 25 (strong trend) AND volume spike
            if (price > r3_val and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S3 AND 1d ADX > 25 (strong trend) AND volume spike
            elif (price < s3_val and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S3 OR ADX < 20 (trend weakening)
                if (price < s3_val or adx_val < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above R3 OR ADX < 20 (trend weakening)
                if (price > r3_val or adx_val < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3_S3_Breakout_1dADX_Volume"
timeframe = "12h"
leverage = 1.0