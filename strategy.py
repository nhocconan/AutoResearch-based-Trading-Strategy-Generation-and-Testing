#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R3/S3) breakout with volume confirmation and ADX trend filter.
# Uses only 3 conditions: price breaks R3/S3, volume > 1.5x average, ADX > 25.
# Works in bull/bear by capturing breakouts with institutional volume in trending markets.
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (using close of previous day)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use previous day's data to avoid look-ahead
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses its own close
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    diff = prev_high - prev_low
    r3 = prev_close + 1.1 * diff
    s3 = prev_close - 1.1 * diff
    r4 = prev_close + 1.5 * diff
    s4 = prev_close - 1.5 * diff
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 14-period ADX on 1d for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # first period
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = np.zeros_like(tr)
        plus_dm_sm = np.zeros_like(plus_dm)
        minus_dm_sm = np.zeros_like(minus_dm)
        
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            plus_dm_sm[period-1] = np.mean(plus_dm[:period])
            minus_dm_sm[period-1] = np.mean(minus_dm[:period])
            
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_sm[i] = (plus_dm_sm[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_sm[i] = (minus_dm_sm[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_sm / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_sm / atr, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        # Smoothed DX (ADX)
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike detection (20-period average on 4h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start after warmup period
        # Skip if any critical data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(adx_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume confirmation and ADX > 25
            if (close[i] > r3_4h[i] and volume_ratio > 1.5 and adx_4h[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S3 with volume confirmation and ADX > 25
            elif (close[i] < s3_4h[i] and volume_ratio > 1.5 and adx_4h[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below S3 or ADX drops below 20 (trend weakening)
            if close[i] < s3_4h[i] or adx_4h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above R3 or ADX drops below 20
            if close[i] > r3_4h[i] or adx_4h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_R3S3_Volume_ADX"
timeframe = "4h"
leverage = 1.0