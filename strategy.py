#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_VolumeFilter_V1
Hypothesis: Use daily Camarilla pivot levels (R1/S1) as support/resistance on 12h timeframe.
Go long when price breaks above R1 with volume confirmation, short when price breaks below S1 with volume confirmation.
Add 1d ADX filter to avoid ranging markets (ADX < 20). Only trade during London/New York session overlap (12-16 UTC).
Target: 15-25 trades/year by requiring multiple confluence factors. Works in bull via breakouts and in bear via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # We need previous day's values, so shift by 1
    if len(close_1d) >= 2:
        prev_close = close_1d[:-1]
        prev_high = high_1d[:-1]
        prev_low = low_1d[:-1]
        # Calculate for each day, then shift forward to align with current day
        camarilla_width = 1.1 * (prev_high - prev_low) / 12.0
        r1_raw = prev_close + camarilla_width
        s1_raw = prev_close - camarilla_width
        # Shift to get today's levels based on yesterday's action
        r1_1d = np.empty_like(close_1d)
        s1_1d = np.empty_like(close_1d)
        r1_1d[0] = np.nan  # No previous day for first bar
        s1_1d[0] = np.nan
        r1_1d[1:] = r1_raw[:-1]
        s1_1d[1:] = s1_raw[:-1]
    else:
        r1_1d = np.full_like(close_1d, np.nan)
        s1_1d = np.full_like(close_1d, np.nan)
    
    # Calculate 14-period ADX for ranging/trending filter
    if len(high_1d) >= 2:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Pad first element
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        # Pad first element
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smooth with Wilder's smoothing (14-period)
        def wilders_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period + 1:
                # First value is simple average
                result[period] = np.nanmean(data[1:period+1])
                # Subsequent values
                for i in range(period + 1, len(data)):
                    if not np.isnan(result[i-1]):
                        result[i] = (result[i-1] * (period - 1) + data[i]) / period
            return result
        
        atr = wilders_smooth(tr, 14)
        plus_di = 100 * wilders_smooth(plus_dm, 14) / np.where(at != 0, atr, np.nan)
        minus_di = 100 * wilders_smooth(minus_dm, 14) / np.where(at != 0, atr, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.nan)
        adx = wilders_smooth(dx, 14)
    else:
        atr = np.full_like(close_1d, np.nan)
        adx = np.full_like(close_1d, np.nan)
    
    # Align 1d indicators to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average on 12h
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Session filter: London/NY overlap (12-16 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1) + 1  # Need volume MA and at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 12 <= hour <= 16  # London/NY overlap
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: only trade when ADX > 20 (trending market)
        trending = adx_1d_aligned[i] > 20
        
        if position == 0 and in_session and vol_confirm and trending:
            # Long: price breaks above R1
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1
            elif close[i] < s1_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal) or ADX drops (losing trend)
            if close[i] < s1_1d_aligned[i] or adx_1d_aligned[i] < 15:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal) or ADX drops (losing trend)
            if close[i] > r1_1d_aligned[i] or adx_1d_aligned[i] < 15:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0