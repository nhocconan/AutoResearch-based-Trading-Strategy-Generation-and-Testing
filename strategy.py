#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Bollinger Bands with 1-week ADX trend filter.
# Long when price touches lower BB in uptrend (ADX>25), short when touches upper BB in downtrend.
# Uses Bollinger Band width to filter ranging markets (BW<0.05 = range, BW>0.08 = trend).
# Designed for low trade frequency (20-40/year) with mean reversion in trends.
# BB touch provides precise entry, ADX ensures trend alignment, BW filter avoids chop.

name = "4h_1dBB_1wADX_TrendMeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_middle = np.zeros_like(close_1d)
    bb_std = np.zeros_like(close_1d)
    bb_upper = np.zeros_like(close_1d)
    bb_lower = np.zeros_like(close_1d)
    
    for i in range(19, len(close_1d)):
        bb_middle[i] = np.mean(close_1d[i-19:i+1])
        bb_std[i] = np.std(close_1d[i-19:i+1])
        bb_upper[i] = bb_middle[i] + 2 * bb_std[i]
        bb_lower[i] = bb_middle[i] - 2 * bb_std[i]
    
    # First 19 days have no data
    bb_middle[:19] = bb_std[:19] = bb_upper[:19] = bb_lower[:19] = np.nan
    
    # Align Bollinger Bands to 4h timeframe
    bb_middle_4h = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_upper_4h = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_4h = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Get weekly data for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0
        
        # Smoothed values
        atr = np.zeros(n)
        plus_dm_smooth = np.zeros(n)
        minus_dm_smooth = np.zeros(n)
        
        # Initial values
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
        
        # Wilder smoothing
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)
        for i in range(period, n):
            if atr[i] != 0:
                plus_di[i] = plus_dm_smooth[i] / atr[i] * 100
                minus_di[i] = minus_dm_smooth[i] / atr[i] * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        # ADX
        adx = np.full(n, np.nan)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_4h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Bollinger Band Width (for regime filter)
    bb_width = (bb_upper_4h - bb_lower_4h) / bb_middle_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_lower_4h[i]) or np.isnan(bb_upper_4h[i]) or 
            np.isnan(bb_middle_4h[i]) or np.isnan(adx_1w_4h[i]) or
            np.isnan(bb_width[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when BB width indicates trend (not range)
        is_trending = bb_width[i] > 0.08
        
        if position == 0:
            # Mean reversion entries in trending markets
            if is_trending:
                # Long when price touches lower BB in uptrend (ADX>25 and price above middle)
                if close[i] <= bb_lower_4h[i] and adx_1w_4h[i] > 25 and close[i] > bb_middle_4h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short when price touches upper BB in downtrend (ADX>25 and price below middle)
                elif close[i] >= bb_upper_4h[i] and adx_1w_4h[i] > 25 and close[i] < bb_middle_4h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to middle BB or trend weakens
            if close[i] >= bb_middle_4h[i] or adx_1w_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle BB or trend weakens
            if close[i] <= bb_middle_4h[i] or adx_1w_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals