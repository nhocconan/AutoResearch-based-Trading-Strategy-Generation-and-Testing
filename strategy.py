#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1d ADX25 trend filter
# Long when price breaks above R3 AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending)
# Short when price breaks below S3 AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending)
# Exit when price crosses back to H3/L3 level OR 1d ADX < 20 (range regime)
# Uses discrete sizing (0.30) to limit fee drag. Target: 20-40 trades/year per symbol.
# Camarilla levels provide intraday support/resistance, volume spike confirms institutional interest,
# 1d ADX filters for trending markets to avoid counter-trend whipsaws in ranging markets.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Camarilla_R3S3_Breakout_1dADX25_Trend_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Camarilla levels and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d data (using previous day's OHLC)
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + (rang * 1.1 / 4)
    camarilla_l3 = prev_close - (rang * 1.1 / 4)
    camarilla_h4 = prev_close + (rang * 1.1 / 2)
    camarilla_l4 = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d ADX for trend filter
    # ADX calculation requires: +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[np.nan], high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    
    # Trend filter: ADX > 25 indicates trending market
    trending = adx > 25
    # Exit filter: ADX < 20 indicates ranging market (hysteresis)
    ranging = adx < 20
    
    # Align 1d indicators to 4h timeframe
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    ranging_aligned = align_htf_to_ltf(prices, df_1d, ranging.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(trending_aligned[i]) or 
            np.isnan(ranging_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3 AND volume spike AND trending market (ADX > 25)
            if (close[i] > h3_aligned[i] and 
                volume_filter[i] and 
                trending_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below S3 AND volume spike AND trending market (ADX > 25)
            elif (close[i] < l3_aligned[i] and 
                  volume_filter[i] and 
                  trending_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back to L3 OR market becomes ranging (ADX < 20)
            if (close[i] < l3_aligned[i] or 
                ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back to H3 OR market becomes ranging (ADX < 20)
            if (close[i] > h3_aligned[i] or 
                ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals