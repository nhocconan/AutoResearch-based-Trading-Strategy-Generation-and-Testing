#!/usr/bin/env python3
# 6h_ADX_Williams_Alligator_Trend_Filter
# Hypothesis: Uses Williams Alligator (13/8/5 SMAs) to identify trending vs ranging markets.
# In trending markets (Alligator jaws open), trades are taken in the direction of ADX (>25).
# Long when price > Alligator teeth and ADX rising; Short when price < Alligator teeth and ADX rising.
# Uses 1d trend filter to avoid counter-trend trades. Designed for 6h timeframe to capture medium-term trends
# with low trade frequency (target: 20-50 trades/year) to minimize fee drag in both bull and bear markets.

name = "6h_ADX_Williams_Alligator_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs on median price
    # Using 1d data as per hypothesis
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = median_price.rolling(window=13, min_periods=13).mean().values  # Alligator Jaw (13-period)
    teeth = median_price.rolling(window=8, min_periods=8).mean().values   # Alligator Teeth (8-period)
    lips = median_price.rolling(window=5, min_periods=5).mean().values    # Alligator Lips (5-period)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # ADX calculation (14-period) on 1d data for trend strength
    # Calculate +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            # Wilder's smoothing: today = (1-1/period)*yesterday + (1/period)*today
            for i in range(period, len(data)):
                result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
        return result
    
    # Calculate smoothed +DM, -DM, TR
    if len(plus_dm) >= 14:
        plus_dm_smoothed = wilders_smooth(plus_dm, 14)
        minus_dm_smoothed = wilders_smooth(minus_dm, 14)
        tr_smoothed = wilders_smooth(tr, 14)
        
        # Avoid division by zero
        plus_di = np.where(tr_smoothed != 0, 100 * plus_dm_smoothed / tr_smoothed, 0)
        minus_di = np.where(tr_smoothed != 0, 100 * minus_dm_smoothed / tr_smoothed, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smooth(dx, 14)
        
        # Align ADX to 6h timeframe (no additional delay needed for ADX)
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # 1d EMA50 for trend filter (price > EMA50 = uptrend bias)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator: jaws open when teeth > lips (for uptrend) or teeth < lips (for downtrend)
        # We require clear separation: teeth and lips not intertwined
        alligator_jaw_open = teeth_aligned[i] > lips_aligned[i]  # Potential uptrend
        alligator_jaw_open_down = teeth_aligned[i] < lips_aligned[i]  # Potential downtrend
        
        # ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Price relative to Alligator teeth (8-period SMA)
        price_vs_teeth = close[i] - teeth_aligned[i]
        
        if position == 0:
            # LONG: Alligator suggests uptrend (teeth > lips), ADX strong, price above teeth, and uptrend bias (price > EMA50)
            if alligator_jaw_open and strong_trend and price_vs_teeth > 0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator suggests downtrend (teeth < lips), ADX strong, price below teeth, and downtrend bias (price < EMA50)
            elif alligator_jaw_open_down and strong_trend and price_vs_teeth < 0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator closes (teeth <= lips) OR ADX weakens (<20) OR price crosses below teeth
            if (teeth_aligned[i] <= lips_aligned[i]) or (adx_aligned[i] < 20) or (close[i] < teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator closes (teeth >= lips) OR ADX weakens (<20) OR price crosses above teeth
            if (teeth_aligned[i] >= lips_aligned[i]) or (adx_aligned[i] < 20) or (close[i] > teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals