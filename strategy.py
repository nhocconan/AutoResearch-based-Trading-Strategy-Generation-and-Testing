#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Camarilla pivot levels from 1d timeframe with breakout/mean-reversion logic. 
Long when price breaks above R1 with volume confirmation in uptrend (price > EMA34).
Short when price breaks below S1 with volume confirmation in downtrend (price < EMA34).
Exit when price reaches opposite S1/R1 level or closes back inside the (R1,S1) range.
ATR filter ensures volatility is sufficient to avoid chop. Designed for 6s timeframe to capture 
intraday swings with low frequency. Works in both bull (breakouts) and bear (mean reversion at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Camarilla pivot, EMA trend, and volume average ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # R2 = close + 0.5 * (high - low)
    # R1 = close + 0.25 * (high - low)
    # S1 = close - 0.25 * (high - low)
    # S2 = close - 0.5 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # We use R1, S1 for breakout and S3, R3 for extreme fade (not used here for simplicity)
    
    # Need previous day's high, low, close to calculate today's pivot
    # Since we're calculating for current bar, we use previous day's values
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day will have rolled values from last - handle with nans
    
    cam_r1 = prev_close_1d + 0.25 * (prev_high_1d - prev_low_1d)
    cam_s1 = prev_close_1d - 0.25 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 6s timeframe (these levels are constant throughout the day)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume average (20-period) for volume filter
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # ATR filter (6s timeframe) - use 14-period ATR
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA34, ATR, and volume average rollouts
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(cam_r1_aligned[i]) or 
            np.isnan(cam_s1_aligned[i]) or 
            np.isnan(vol_avg20_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        # ATR filter: ensure volatility is sufficient (avoid choppy markets)
        atr_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Combined filter
        filter_ok = vol_filter and atr_filter
        
        # Entry conditions
        if position == 0:
            # Long: break above R1 in uptrend (close > EMA34) with filter
            if close[i] > cam_r1_aligned[i] and close[i] > ema34_1d_aligned[i] and filter_ok:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 in downtrend (close < EMA34) with filter
            elif close[i] < cam_s1_aligned[i] and close[i] < ema34_1d_aligned[i] and filter_ok:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions
        elif position == 1:
            # Exit long when price reaches S1 (opposite level) or closes back below R1
            if close[i] <= cam_s1_aligned[i] or close[i] < cam_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price reaches R1 (opposite level) or closes back above S1
            if close[i] >= cam_r1_aligned[i] or close[i] > cam_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0