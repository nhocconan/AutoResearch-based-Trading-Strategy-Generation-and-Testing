#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v9
Hypothesis: Further tighten entry conditions to reduce overtrading. Use EMA100 trend filter (more stable), require volume spike >8.0x average, and add choppiness regime filter (CHOP > 61.8 = ranging) to avoid false breakouts in chop. Target 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA100 for trend filter (more stable than EMA50)
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 4h volume ratio (current vs 30-period average = 7.5h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    def calculate_chop(high, low, close, window=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        for i in range(window, len(close)):
            atr[i] = np.mean(tr[i-window+1:i+1])
        # Avoid division by zero
        max_high = np.zeros(len(close))
        min_low = np.zeros(len(close))
        for i in range(window, len(close)):
            max_high[i] = np.max(high[i-window+1:i+1])
            min_low[i] = np.min(low[i-window+1:i+1])
        chop = np.zeros(len(close))
        for i in range(window, len(close)):
            if max_high[i] - min_low[i] > 0:
                chop[i] = 100 * np.log10(sum(tr[i-window+1:i+1]) / np.log(window) / (max_high[i] - min_low[i]))
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume MA and CHOP
    start_idx = max(100, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_100_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Determine 1d trend (bullish = price above EMA100)
        df_1d_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(df_1d_close_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        htf_1d_bullish = df_1d_close_aligned[i] > ema_100_1d_aligned[i]
        htf_1d_bearish = df_1d_close_aligned[i] < ema_100_1d_aligned[i]
        
        # Volume confirmation: need significant spike (vol_ratio > 8.0) - much stricter than v8
        volume_confirmed = vol_ratio[i] > 8.0
        
        # Choppiness regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8 if not np.isnan(chop[i]) else False
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation + chop filter
            long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_1d_bullish and volume_confirmed and chop_filter
            
            # Short setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation + chop filter
            short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_1d_bearish and volume_confirmed and chop_filter
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Camarilla S1 (opposite level) OR 1d trend turns bearish
            if (close[i] <= camarilla_s1_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (opposite level) OR 1d trend turns bullish
            if (close[i] >= camarilla_r1_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v9"
timeframe = "4h"
leverage = 1.0