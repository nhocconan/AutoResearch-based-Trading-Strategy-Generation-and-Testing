#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA50_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakouts with 1d EMA50 trend filter, volume confirmation (>2.0x 20-bar MA), and 4h chop regime filter (CHOP > 61.8 = range). Uses 1d EMA50 for smoother trend identification and tighter volume confirmation to reduce false breakouts. Targets 20-40 trades/year by requiring confluence of trend, structure, volume, and regime conditions. Works in bull/bear markets by following 1d trend while using Camarilla structure for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter (smoother than EMA34)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's OHLC for Camarilla levels (R3/S3 = wider breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (wider breakout levels for stronger momentum)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (dynamic threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 4h Chop regime filter: avoid breakouts in ranging markets (CHOP > 61.8 = choppy)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index"""
        atr = []
        for i in range(len(high_arr)):
            if i == 0:
                tr = high_arr[i] - low_arr[i]
            else:
                tr = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i-1]), abs(low_arr[i] - close_arr[i-1]))
            atr.append(tr)
        
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        chop = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if atr_sum[i] > 0 and hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(window)
            else:
                chop[i] = 50.0  # neutral when undefined
        return chop
    
    chop_values = calculate_chop(high, low, close, window=14)
    chop_regime = chop_values > 61.8  # True when choppy/ranging (avoid breakouts)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 50 for 1d EMA, 14 for chop)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop_values[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        is_choppy = chop_regime[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla R3/S3 in trend direction with volume spike AND not in choppy regime
        long_entry = (close_val > camarilla_r3_val) and bullish_1d and vol_spike and (not is_choppy)
        short_entry = (close_val < camarilla_s3_val) and bearish_1d and vol_spike and (not is_choppy)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to mid-point or trend change or chop regime
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val < mid_point or not bullish_1d or is_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to mid-point or trend change or chop regime
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val > mid_point or not bearish_1d or is_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA50_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0