#!/usr/bin/env python3
"""
12h_1d_Alligator_Trend_v1
Hypothesis: 12h Williams Alligator (13,8,5 SMAs with future shift) + 12h price above/below teeth (8) + volume confirmation (2x average) + ADX(14) > 20 for trend strength. Exit when price crosses back below/above teeth. Designed for low trade frequency (15-30/year) by requiring strong trend alignment and volume surge. Works in bull/bear via Alligator convergence/divergence and ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Alligator_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR ADX ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily timeframe
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0.0], tr])  # align length
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smooth TR and DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # === 12H DATA FOR ALLIGATOR ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Williams Alligator: Jaw (13 SMA, 8 bars future), Teeth (8 SMA, 5 bars future), Lips (5 SMA, 3 bars future)
    def sma_with_shift(data, period, shift):
        sma = np.full_like(data, np.nan)
        if len(data) >= period:
            for i in range(period-1, len(data)):
                sma[i] = np.mean(data[i-period+1:i+1])
        # Shift forward (future values)
        shifted = np.full_like(sma, np.nan)
        if len(sma) > shift:
            shifted[:-shift] = sma[shift:]
        return shifted
    
    jaw = sma_with_shift(close_12h, 13, 8)   # Blue line
    teeth = sma_with_shift(close_12h, 8, 5)   # Red line
    lips = sma_with_shift(close_12h, 5, 3)    # Green line
    
    # Align daily and 12h data to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume average (20-period for 12h = ~10 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2.0x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Alligator conditions: Mouth open (jaw, teeth, lips separated) and aligned
        # Bullish: lips > teeth > jaw (green > red > blue)
        # Bearish: lips < teeth < jaw (green < red < blue)
        bullish_aligned = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_aligned = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Trend strength: ADX > 20
        strong_trend = adx_aligned[i] > 20
        
        # Entry conditions
        long_setup = bullish_aligned and strong_trend and vol_confirm
        short_setup = bearish_aligned and strong_trend and vol_confirm
        
        # Exit when price crosses back below/above teeth (or Alligator starts to close)
        exit_long = close[i] < teeth_aligned[i] or not bullish_aligned
        exit_short = close[i] > teeth_aligned[i] or not bearish_aligned
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals