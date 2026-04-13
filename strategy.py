#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout + 12h volume confirmation + 1d ADX regime
    # Long: Close > R4 AND 12h volume > 1.5x 20-period average AND 1d ADX > 20
    # Short: Close < S4 AND 12h volume > 1.5x 20-period average AND 1d ADX > 20
    # Exit: Close retreats to H3/L3 levels OR ADX < 15 (trend exhaustion)
    # Using Camarilla from 1d for pivot levels, 12h for volume confirmation, 1d for ADX regime
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    
    # Camarilla levels based on previous day
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    r2_1d = pivot_1d + (range_1d * 1.1 / 6)
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    r4_1d = pivot_1d + (range_1d * 1.1 / 2)
    
    # Support levels
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    s2_1d = pivot_1d - (range_1d * 1.1 / 6)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # H3/L3 for exit (closer to pivot)
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 6h (wait for completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Get 12h data for volume confirmation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    # 20-period average volume on 12h
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Get 1d data for ADX regime (call ONCE before loop)
    high_1d_adx = df_1d['high'].values
    low_1d_adx = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d_adx[1:] - low_1d_adx[1:])
    tr2 = np.abs(high_1d_adx[1:] - close_1d_adx[:-1])
    tr3 = np.abs(low_1d_adx[1:] - close_1d_adx[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = high_1d_adx[1:] - high_1d_adx[:-1]
    down_move = low_1d_adx[:-1] - low_1d_adx[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d_adx = wilders_smoothing(tr, 14)
    plus_di_1d_adx = 100 * wilders_smoothing(plus_dm, 14) / atr_1d_adx
    minus_di_1d_adx = 100 * wilders_smoothing(minus_dm, 14) / atr_1d_adx
    dx_1d_adx = 100 * np.abs(plus_di_1d_adx - minus_di_1d_adx) / (plus_di_1d_adx + minus_di_1d_adx)
    adx_1d_adx = wilders_smoothing(dx_1d_adx, 14)
    
    # Align 1d ADX to 6h
    adx_1d_adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(adx_1d_adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_12h_aligned[i]
        
        # Regime filter: only trade when 1d ADX > 20 (trending market)
        trending = adx_1d_adx_aligned[i] > 20
        weak_trend = adx_1d_adx_aligned[i] < 15  # exit condition
        
        # Entry logic: Camarilla breakout + volume + trend
        long_entry = (close[i] > r4_1d_aligned[i]) and volume_confirmed and trending
        short_entry = (close[i] < s4_1d_aligned[i]) and volume_confirmed and trending
        
        # Exit logic: Retreat to H3/L3 OR trend exhaustion
        long_exit = (close[i] < h3_1d_aligned[i]) or weak_trend
        short_exit = (close[i] > l3_1d_aligned[i]) or weak_trend
        
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

name = "6h_12h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "6h"
leverage = 1.0