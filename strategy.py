#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Strict
Strategy: 4h Camarilla pivot R1/S1 breakout with volume confirmation and ADX trend filter.
Long: Price breaks above R1 with volume spike in uptrend (ADX > 25).
Short: Price breaks below S1 with volume spike in downtrend (ADX > 25).
Uses 1D Camarilla levels for structure, volume spike for confirmation, ADX for regime.
Designed for 4h timeframe: ~20-30 trades/year per symbol (80-120 total over 4 years).
Works in bull/bear via ADX trend filter - only trades in trending regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    s1 = close_1d - rng * 1.1 / 12
    
    # ADX calculation for trend filter (14-period)
    # +DM, -DM, TR
    high_1440 = high  # 4h data, but we need daily equivalent for ADX
    low_1440 = low
    close_1440 = close
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Calculate +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smooth(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            # First value is simple average
            result[period-1] = np.nansum(x[:period]) / period
            # Subsequent values: smoothed = prev_smooth - (prev_smooth/period) + current
            for i in range(period, len(x)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] - (result[i-1]/period) + x[i]
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    
    # Align Camarilla levels and ADX to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above R1
            if trending and vol_confirm and breakout_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakout below S1
            elif trending and vol_confirm and breakout_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend ends, volume confirmation, or breakdown below S1
            if not trending or vol_confirm or close[i] < s1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend ends, volume confirmation, or breakout above R1
            if not trending or vol_confirm or close[i] > r1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Strict"
timeframe = "4h"
leverage = 1.0