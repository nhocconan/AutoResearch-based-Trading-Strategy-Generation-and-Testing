#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d ADX trend filter and volume confirmation
# Long when: price breaks above R4, volume > 2.0x 20-period average, and 1d ADX > 25 (trending)
# Short when: price breaks below S4, volume > 2.0x 20-period average, and 1d ADX > 25
# Exit when price returns to Camarilla R4/S4 level (mean reversion)
# Uses Camarilla levels from 1d for structure, ADX for trend filtering to avoid whipsaws in ranging markets.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R4S4_Breakout_1dADX25_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Camarilla levels and ADX trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) for trend filter
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
        tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # Align with original indices
        
        # Directional Movement
        up_move = np.diff(high_1d)
        down_move = -np.diff(low_1d)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period])
                for i in range(period, len(data)):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        tr14 = WilderSmooth(tr, 14)
        plus_dm14 = WilderSmooth(plus_dm, 14)
        minus_dm14 = WilderSmooth(minus_dm, 14)
        
        # DI+ and DI-
        plus_di14 = 100 * plus_dm14 / tr14
        minus_di14 = 100 * minus_dm14 / tr14
        
        # DX and ADX
        dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
        adx = WilderSmooth(dx, 14)
    else:
        adx = np.full(len(close_1d), np.nan)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels from previous 1d bar
    if len(high_1d) >= 2:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r4 = prev_close + 1.1 * rang * 1.5 / 4
        camarilla_s4 = prev_close - 1.1 * rang * 1.5 / 4
    else:
        camarilla_r4 = np.full(len(close_1d), np.nan)
        camarilla_s4 = np.full(len(close_1d), np.nan)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4, volume filter, and ADX > 25 (trending)
            if (close[i] > camarilla_r4_aligned[i] and 
                open_price[i] <= camarilla_r4_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S4, volume filter, and ADX > 25 (trending)
            elif (close[i] < camarilla_s4_aligned[i] and 
                  open_price[i] >= camarilla_s4_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R4 (mean reversion)
            if close[i] < camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S4 (mean reversion)
            if close[i] > camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals