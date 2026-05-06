#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator uses smoothed medians (Jaw/Teeth/Lips) to identify trending vs ranging markets
# In trending markets (Lips > Teeth > Jaw for uptrend, reverse for downtrend), we trade breakouts
# In ranging markets (Alligator lines intertwined), we fade extremes at 1d Camarilla R3/S3 levels
# Volume spike (>2x 20-bar average) confirms institutional participation
# Discrete sizing 0.25 to limit fee drag; target 50-150 trades over 4 years
# Alligator is particularly effective in crypto markets due to its smoothing reducing whipsaw

name = "6h_WilliamsAlligator_1dEMA34_Camarilla_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams Alligator (smoothed medians)
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.empty_like(source)
        result[:] = np.nan
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    median_1d = (high_1d + low_1d) / 2.0
    jaw = smma(smma(median_1d, 13), 8)
    teeth = smma(smma(median_1d, 8), 5)
    lips = smma(smma(median_1d, 5), 3)
    
    # Calculate 1d Camarilla levels from prior 1d bar
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    pivot_1d = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4.0  # R3
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4.0  # S3
    
    # Calculate 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime using Alligator
        # Trending up: Lips > Teeth > Jaw
        # Trending down: Lips < Teeth < Jaw
        # Ranging: otherwise (lines intertwined)
        is_trending_up = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        is_trending_down = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        is_ranging = not (is_trending_up or is_trending_down)
        
        if position == 0:
            if is_trending_up:
                # In uptrend: buy breakout above Camarilla R3 with volume
                if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_trending_down:
                # In downtrend: sell breakdown below Camarilla S3 with volume
                if close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # ranging market
                # In range: fade at extremes (Camarilla S3/R3) with volume
                if close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25  # long at support
                    position = 1
                elif close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25  # short at resistance
                    position = -1
        elif position == 1:
            # Exit long: price reaches opposite extreme or Alligator reverses
            if is_trending_down or close[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches opposite extreme or Alligator reverses
            if is_trending_up or close[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals