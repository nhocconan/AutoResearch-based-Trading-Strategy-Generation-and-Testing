#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1d ADX25 trend filter
# Long when price breaks above R3 AND volume > 2.0x 20-period average AND 1d ADX > 25 (trending market)
# Short when price breaks below S3 AND volume > 2.0x 20-period average AND 1d ADX > 25 (trending market)
# Exit when price crosses back to H3/L3 level OR 1d ADX drops below 20 (range market)
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Camarilla levels provide intraday support/resistance, volume spike confirms institutional interest,
# 1d ADX filter avoids whipsaws in ranging markets while capturing strong trends.
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
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = np.concatenate([[0], plus_dm])  # Align with original arrays
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilders_smoothing(values, period):
        """Wilder's smoothing: first value is SMA, then recursive EMA"""
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]) and not np.isnan(values[i]):
                result[i] = result[i-1] + (values[i] - result[i-1]) / period
            else:
                result[i] = np.nan
        return result
    
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI values
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Trend filter: ADX > 25 indicates trending market
    trending = adx > 25
    # Exit filter: ADX < 20 indicates ranging market (hysteresis)
    ranging = adx < 20
    
    # Align 1d indicators to 4h timeframe
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    ranging_aligned = align_htf_to_ltf(prices, df_1d, ranging.astype(float))
    
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
            # Long conditions: price breaks above H3 AND volume spike AND trending market
            if (close[i] > h3_aligned[i] and 
                volume_filter[i] and 
                trending_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below L3 AND volume spike AND trending market
            elif (close[i] < l3_aligned[i] and 
                  volume_filter[i] and 
                  trending_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to L3 OR market becomes ranging (ADX < 20)
            if (close[i] < l3_aligned[i] or 
                ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to H3 OR market becomes ranging (ADX < 20)
            if (close[i] > h3_aligned[i] or 
                ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals