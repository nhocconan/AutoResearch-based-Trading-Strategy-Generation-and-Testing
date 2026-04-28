#!/usr/bin/env python3
"""
1h_4hTrend_DMI_Filter
Hypothesis: Uses 4h ADX > 25 as trend filter, 1h +DI/-DI crossovers for entries.
Only trades in strong trends (ADX > 25) to avoid whipsaws in ranging markets.
Designed for low trade frequency (<30/year) to minimize fee burn while capturing
strong directional moves. Works in both bull and bear markets by following trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (EMA alpha = 1/period)
        alpha = 1.0 / period
        atr = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(plus_dm, np.nan)
        minus_dm_smooth = np.full_like(minus_dm, np.nan)
        
        # Initialize first values
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        # Wilder smoothing
        for i in range(period + 1, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        # Calculate DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (smoothed DX)
        adx = np.full_like(dx, np.nan)
        adx[2*period] = np.nansum(dx[period+1:2*period+1])
        for i in range(2*period + 1, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
        
        return adx, plus_di, minus_di
    
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Align to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    plus_di_4h_aligned = align_htf_to_ltf(prices, df_4h, plus_di_4h)
    minus_di_4h_aligned = align_htf_to_ltf(prices, df_4h, minus_di_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(adx_4h_aligned[i]) or 
            np.isnan(plus_di_4h_aligned[i]) or
            np.isnan(minus_di_4h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_4h_aligned[i] > 25
        
        # Entry signals: DI crossovers
        bullish_cross = plus_di_4h_aligned[i] > minus_di_4h_aligned[i]
        bearish_cross = plus_di_4h_aligned[i] < minus_di_4h_aligned[i]
        
        # Exit signals: trend weakening or reverse crossover
        trend_weakening = adx_4h_aligned[i] < 20
        reverse_cross_bull = plus_di_4h_aligned[i] < minus_di_4h_aligned[i]
        reverse_cross_bear = plus_di_4h_aligned[i] > minus_di_4h_aligned[i]
        
        if strong_trend and bullish_cross and position <= 0:
            signals[i] = 0.20
            position = 1
        elif strong_trend and bearish_cross and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (trend_weakening or reverse_cross_bull) and position == 1:
            signals[i] = 0.0
            position = 0
        elif (trend_weakening or reverse_cross_bear) and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4hTrend_DMI_Filter"
timeframe = "1h"
leverage = 1.0