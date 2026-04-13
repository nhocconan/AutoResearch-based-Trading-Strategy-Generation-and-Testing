#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + ADX trend filter
    # Long: price breaks above H3 (1d) AND volume > 1.5x 20-period average AND ADX(14) > 25
    # Short: price breaks below L3 (1d) AND volume > 1.5x 20-period average AND ADX(14) > 25
    # Exit: price returns to Pivot Point (PP) or volume drops below average
    # Using 12h timeframe for low trade frequency, Camarilla from 1d for structure,
    # volume for confirmation, ADX for trend strength (avoid chop).
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations (based on previous day)
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r4 = pp + range_1d * 1.1 / 2
    r3 = pp + range_1d * 1.1 / 4
    r2 = pp + range_1d * 1.1 / 6
    r1 = pp + range_1d * 1.1 / 12
    
    # Support levels
    s1 = pp - range_1d * 1.1 / 12
    s2 = pp - range_1d * 1.1 / 6
    s3 = pp - range_1d * 1.1 / 4
    s4 = pp - range_1d * 1.1 / 2
    
    # Align daily Camarilla levels to 12h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate daily ADX(14) for trend filter
    # ADX requires +DI, -DI, and TR
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values (Wilder's smoothing)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 = strong trend
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + trend + volume
        long_entry = (close[i] > r3_1d_aligned[i]) and trending and vol_confirm
        short_entry = (close[i] < s3_1d_aligned[i]) and trending and vol_confirm
        
        # Exit logic: price returns to PP or volume drops
        long_exit = (close[i] < pp_1d_aligned[i]) or not vol_confirm
        short_exit = (close[i] > pp_1d_aligned[i]) or not vol_confirm
        
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

name = "12h_1d_camarilla_breakout_adx_volume_v1"
timeframe = "12h"
leverage = 1.0