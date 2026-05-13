#!/usr/bin/env python3
"""
1D_WeeklyDonchian_Breakout_RangeFilter
Hypothesis: Weekly Donchian channels capture the primary trend, while daily ADX and volatility filters ensure we only trade in strong trends, avoiding whipsaws in ranging markets. Works in bull markets by catching breakouts and in bear markets by catching breakdowns with volume confirmation.
"""

name = "1D_WeeklyDonchian_Breakout_RangeFilter"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate 20-period weekly Donchian channels
    donchian_period = 20
    upper_donch = np.full_like(close_weekly, np.nan)
    lower_donch = np.full_like(close_weekly, np.nan)
    
    for i in range(donchian_period - 1, len(close_weekly)):
        upper_donch[i] = np.max(high_weekly[i-donchian_period+1:i+1])
        lower_donch[i] = np.min(low_weekly[i-donchian_period+1:i+1])
    
    # Align weekly Donchian to daily timeframe
    upper_donch_aligned = align_htf_to_ltf(prices, df_weekly, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_weekly, lower_donch)
    
    # Calculate daily ADX for trend strength filter
    period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr = np.zeros_like(close)
    plus_dm_smooth = np.zeros_like(close)
    minus_dm_smooth = np.zeros_like(close)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    # Wilder's smoothing
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Calculate +DI and -DI
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(close)
    
    # Initial ADX value
    if len(dx) >= 2*period-1:
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Volume average for confirmation
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # ADX threshold for strong trend
        strong_trend = adx[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian upper + strong trend + volume
            if close[i] > upper_donch_aligned[i] and strong_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian lower + strong trend + volume
            elif close[i] < lower_donch_aligned[i] and strong_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian lower or trend weakens
            if close[i] < lower_donch_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian upper or trend weakens
            if close[i] > upper_donch_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals