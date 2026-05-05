#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d ADX trend filter and volume spike confirmation
# Long when price breaks above upper BB(20,2) AND volume > 2.0x 20-period average AND 1d ADX > 25 (trending)
# Short when price breaks below lower BB(20,2) AND volume > 2.0x 20-period average AND 1d ADX > 25 (trending)
# Exit when price crosses middle BB (20-period SMA) OR 1d ADX drops below 20 (range)
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-35 trades/year per symbol.
# Bollinger Bands provide dynamic support/resistance, volume spike confirms institutional participation,
# 1d ADX filter ensures we only trade in trending markets to avoid choppy whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_BB_Breakout_ADXTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on 4h close
    close_4h = df_4h['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    middle_bb = sma_20  # 20-period SMA
    
    # Align Bollinger Bands to prices timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_4h, middle_bb)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    if len(tr) >= 14:
        atr_14 = WilderSmooth(tr, 14)
        dm_plus_smooth = WilderSmooth(dm_plus, 14)
        dm_minus_smooth = WilderSmooth(dm_minus, 14)
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / atr_14
        di_minus = 100 * dm_minus_smooth / atr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx = np.where(np.isnan(dx), 0, dx)
        adx = WilderSmooth(dx, 14)
        
        # Trend filter: ADX > 25 = trending, ADX < 20 = ranging
        adx_trending = adx > 25
        adx_ranging = adx < 20
    else:
        adx_trending = np.zeros(len(close_1d), dtype=bool)
        adx_ranging = np.ones(len(close_1d), dtype=bool)
    
    # Align 1d ADX to 4h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or 
            np.isnan(adx_trending_aligned[i]) or 
            np.isnan(adx_ranging_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper BB AND volume spike AND 1d ADX trending
            if (close[i] > upper_bb_aligned[i] and 
                volume_filter[i] and 
                adx_trending_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB AND volume spike AND 1d ADX trending
            elif (close[i] < lower_bb_aligned[i] and 
                  volume_filter[i] and 
                  adx_trending_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle BB OR 1d ADX becomes ranging
            if (close[i] < middle_bb_aligned[i] or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle BB OR 1d ADX becomes ranging
            if (close[i] > middle_bb_aligned[i] or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals