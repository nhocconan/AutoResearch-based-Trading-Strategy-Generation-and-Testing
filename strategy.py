#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaws 13, Teeth 8, Lips 5) to identify trends.
# In uptrend: Lips > Teeth > Jaws; in downtrend: Jaws > Teeth > Lips.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation reduces false signals. Designed for low trade frequency
# on 12h timeframe to avoid fee drag while capturing major trends.
# Works in bull/bear by only taking signals in direction of weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get daily data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator: three SMAs based on median price
    median_price = (high_1d + low_1d) / 2
    
    # Jaws: 13-period SMMA, shifted 8 bars
    jaws = np.full(len(df_1d), np.nan)
    # Teeth: 8-period SMMA, shifted 5 bars  
    teeth = np.full(len(df_1d), np.nan)
    # Lips: 5-period SMMA, shifted 3 bars
    lips = np.full(len(df_1d), np.nan)
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts: Jaws shifted 8, Teeth shifted 5, Lips shifted 3
    for i in range(len(jaws_raw)):
        if i + 8 < len(jaws):
            jaws[i + 8] = jaws_raw[i]
        if i + 5 < len(teeth):
            teeth[i + 5] = teeth_raw[i]
        if i + 3 < len(lips):
            lips[i + 3] = lips_raw[i]
    
    # Align Alligator lines to 12h timeframe (wait for daily close)
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Weekly trend filter: 50-period EMA
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 1.5 x 28-period average (14 days of 12h bars)
    vol_ma_28 = np.full(n, np.nan)
    for i in range(27, n):
        vol_ma_28[i] = np.mean(volume[i-27:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily data (13+8=21 for jaws), weekly EMA (50), volume MA (28)
    start_idx = max(21, 50, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_28[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_28[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        # Williams Alligator signals
        # Bullish: Lips > Teeth > Jaws (all aligned upward)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]
        # Bearish: Jaws > Teeth > Lips (all aligned downward)
        bearish_aligned = jaws_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: bullish alignment with volume and weekly uptrend
            if bullish_aligned and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: bearish alignment with volume and weekly downtrend
            elif bearish_aligned and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or weekly trend turns bearish
            if bearish_aligned or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish alignment or weekly trend turns bullish
            if bullish_aligned or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Williams_Alligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0