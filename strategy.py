#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Trend
Strategy: 4h Camarilla R1/S1 breakout + volume confirmation + ADX trend filter.
Long: Price breaks above R1 + volume > 1.5x 20-bar avg + ADX > 25 (trending)
Short: Price breaks below S1 + volume > 1.5x 20-bar avg + ADX > 25 (trending)
Exit: Price crosses back through pivot point (mean reversion within trend)
Position size: 0.25
Uses 4h Camarilla levels from 1d OHLC, volume for confirmation, ADX for trend strength.
Avoids range-bound whipsaws by requiring trending conditions (ADX > 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    # R2 = close + 0.6*(high-low), R1 = close + 0.318*(high-low)
    # PP = (high+low+close)/3, S1 = close - 0.318*(high-low)
    # S2 = close - 0.6*(high-low), S3 = close - 1.1*(high-low), 
    # S4 = close - 1.5*(high-low)
    hl_range = high_1d - low_1d
    r1 = close_1d + 0.318 * hl_range
    s1 = close_1d - 0.318 * hl_range
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate ADX(14) on 1d for trend strength
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    # +DM14 = smoothed +DM, -DM14 = smoothed -DM, TR14 = smoothed TR
    # +DI14 = 100 * +DM14 / TR14, -DI14 = 100 * -DM14 / TR14
    # DX = 100 * abs(+DI14 - -DI14) / (+DI14 + -DI14)
    # ADX = smoothed DX
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    plus_dm = np.where((high_1d - prev_high) > (prev_low - low_1d), np.maximum(high_1d - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low_1d) > (high_1d - prev_high), np.maximum(prev_low - low_1d, 0), 0)
    tr = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - prev_close)), np.abs(low_1d - prev_close))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smooth(dx, 14)
    
    # Align Camarilla levels and ADX to lower timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(30, n):  # warmup for ADX(14) and Camarilla
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume aligned to lower timeframe
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > (1.5 * volume_ma20_1d_aligned[i])
        trend_filter = adx_aligned[i] > 25  # trending market
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        # Exit conditions: price crosses pivot point
        exit_long = close[i] < pp_aligned[i]
        exit_short = close[i] > pp_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume + trend
            if breakout_up and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume + trend
            elif breakout_down and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below pivot point
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot point
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0