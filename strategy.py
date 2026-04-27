#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter
Hypothesis: 4h strategy using Camarilla R3/S3 levels from 1d for breakout entries with 1d EMA34 trend filter and choppiness regime filter. 
Only trade when market is trending (CHOP < 38.2) to avoid whipsaws in ranging markets. 
Volume confirmation (>1.8x 20-period average) ensures institutional participation. 
Discrete position sizing (0.25) to minimize fee drift. Designed for low trade frequency (<40/year) with Sharpe > 0 in both bull and bear regimes.
"""

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
    
    # Get 1d data for Camarilla levels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 from 1d OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 2
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 2
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(c_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d Choppiness Index (CHOP) for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index: values > 61.8 = ranging, < 38.2 = trending"""
        atr_arr = np.zeros(len(close_arr))
        tr_arr = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = np.abs(high_arr[i] - close_arr[i-1])
            lc = np.abs(low_arr[i] - close_arr[i-1])
            tr_arr[i] = max(hl, hc, lc)
            atr_arr[i] = atr_arr[i-1] + (tr_arr[i] - atr_arr[i-1]) / period if i >= period else tr_arr[i]
        # Smoothed ATR (using Wilder's smoothing)
        atr_smoothed = np.zeros(len(close_arr))
        atr_smoothed[period] = np.mean(tr_arr[1:period+1]) if period < len(tr_arr) else 0
        for i in range(period+1, len(close_arr)):
            atr_smoothed[i] = (atr_smoothed[i-1] * (period-1) + tr_arr[i]) / period
        # CHOP = 100 * log10(sum(ATR) / (max(H) - min(L))) / log10(period)
        chop_arr = np.full(len(close_arr), 50.0)
        for i in range(period, len(close_arr)):
            sum_atr = np.sum(atr_smoothed[i-period+1:i+1])
            max_h = np.max(high_arr[i-period+1:i+1])
            min_l = np.min(low_arr[i-period+1:i+1])
            if max_h > min_l:
                chop_arr[i] = 100 * np.log10(sum_atr / (max_h - min_l)) / np.log10(period)
        return chop_arr
    
    chop_1d = calculate_chop(h_1d, l_1d, c_1d, 14)
    chop_trending = chop_1d < 38.2  # Trending regime
    
    # Align 1d indicators to 4h timeframe (completed 1d bars only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA34 (34) + volume avg (20) + chop (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_34_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Only trade in trending regime (CHOP < 38.2)
        if not chop_val:
            # Force flat if not trending
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with EMA trend filter and volume confirmation
            # Long: price closes above R3 AND above EMA34 (uptrend)
            long_condition = (close_val > r3_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S3 AND below EMA34 (downtrend)
            short_condition = (close_val < s3_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S3 (opposite level) OR EMA34 turns bearish (price below EMA)
            if (close_val < s3_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (opposite level) OR EMA34 turns bullish (price above EMA)
            if (close_val > r3_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter"
timeframe = "4h"
leverage = 1.0