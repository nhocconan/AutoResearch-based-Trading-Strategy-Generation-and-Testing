#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation
# Uses Camarilla pivot levels (S3/S4 for long, R3/R4 for short) from 12h for reversal signals
# 1d EMA (50) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading reversals against intraday extremes
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formula: 
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    cam_r4 = close_12h + (high_12h - low_12h) * 1.1 / 2
    cam_r3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    cam_s3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    cam_s4 = close_12h - (high_12h - low_12h) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    cam_r4_aligned = align_htf_to_ltf(prices, df_12h, cam_r4)
    cam_r3_aligned = align_htf_to_ltf(prices, df_12h, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_12h, cam_s3)
    cam_s4_aligned = align_htf_to_ltf(prices, df_12h, cam_s4)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(cam_r4_aligned[i]) or np.isnan(cam_r3_aligned[i]) or 
            np.isnan(cam_s3_aligned[i]) or np.isnan(cam_s4_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade against the 1d EMA extreme (mean reversion)
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long: price touches S3/S4 (oversold) with volume filter and below 1d EMA
            if (price <= cam_s3_aligned[i] or price <= cam_s4_aligned[i]) and vol > 1.5 * avg_vol[i] and not above_ema:
                position = 1
                signals[i] = position_size
            # Short: price touches R3/R4 (overbought) with volume filter and above 1d EMA
            elif (price >= cam_r3_aligned[i] or price >= cam_r4_aligned[i]) and vol > 1.5 * avg_vol[i] and above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above the 12h midpoint (mean reversion complete) or above 1d EMA
            mid_point = (cam_s3_aligned[i] + cam_r3_aligned[i]) / 2
            if price > mid_point or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below the 12h midpoint (mean reversion complete) or below 1d EMA
            mid_point = (cam_s3_aligned[i] + cam_r3_aligned[i]) / 2
            if price < mid_point or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0