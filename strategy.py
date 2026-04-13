#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime filter
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    # Trend filter: ADX(14) > 25 on 1d timeframe
    # Entry: Long when Bull Power > 0 AND ADX > 25 AND Bull Power rising
    #        Short when Bear Power > 0 AND ADX > 25 AND Bear Power rising
    # Exit: Opposite power signal or ADX < 20 (regime change)
    # Discrete sizing 0.25 to minimize fee churn. Works in bull/bear via ADX regime.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = np.full(n, np.nan)
    if n >= 13:
        ema_13[12] = np.mean(close[:13])
        multiplier = 2 / (13 + 1)
        for i in range(13, n):
            ema_13[i] = (close[i] - ema_13[i-1]) * multiplier + ema_13[i-1]
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Power rising/falling (1-bar momentum)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20  # exit regime
        
        # Entry conditions
        long_entry = (bull_power[i] > 0) and trending and bull_power_rising[i]
        short_entry = (bear_power[i] > 0) and trending and bear_power_rising[i]
        
        # Exit conditions
        long_exit = (bull_power[i] <= 0) or ranging
        short_exit = (bear_power[i] <= 0) or ranging
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0