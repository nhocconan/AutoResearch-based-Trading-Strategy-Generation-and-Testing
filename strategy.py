#!/usr/bin/env python3
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
    
    # Load 12h data for trend and weekly pivot calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = np.full_like(close_12h, np.nan)
    donch_low = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 20:
        for i in range(19, len(close_12h)):
            donch_high[i] = np.max(high_12h[i-19:i+1])
            donch_low[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate 12h EMA(50) for trend filter
    ema_50 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50[i] = (close_12h[i] * 2 + ema_50[i-1] * 49) / 51  # EMA formula
    
    # Calculate weekly pivot points from 12h data (approximate weekly from 5x 12h bars)
    # Group 12h data into pseudo-weekly (5 periods ≈ 60h ≈ 2.5 days, close enough for weekly pivot)
    weekly_period = 5
    if len(close_12h) >= weekly_period:
        weekly_high = np.full_like(close_12h, np.nan)
        weekly_low = np.full_like(close_12h, np.nan)
        weekly_close = np.full_like(close_12h, np.nan)
        
        for i in range(weekly_period-1, len(close_12h)):
            start_idx = i - weekly_period + 1
            weekly_high[i] = np.max(high_12h[start_idx:i+1])
            weekly_low[i] = np.min(low_12h[start_idx:i+1])
            weekly_close[i] = close_12h[i]  # use last close of period
        
        # Calculate pivot levels from weekly data
        wp = (weekly_high + weekly_low + weekly_close) / 3.0
        wr3 = wp + 2 * (weekly_high - weekly_low)  # R3 equivalent
        ws3 = wp - 2 * (weekly_high - weekly_low)  # S3 equivalent
        wr4 = wp + 3 * (weekly_high - weekly_low)  # R4 equivalent
        ws4 = wp - 3 * (weekly_high - weekly_low)  # S4 equivalent
        
        # Align weekly pivot levels to 12h timeframe
        wp_12h = align_htf_to_ltf(prices, df_12h, wp)
        wr3_12h = align_htf_to_ltf(prices, df_12h, wr3)
        ws3_12h = align_htf_to_ltf(prices, df_12h, ws3)
        wr4_12h = align_htf_to_ltf(prices, df_12h, wr4)
        ws4_12h = align_htf_to_ltf(prices, df_12h, ws4)
    else:
        wp_12h = wr3_12h = ws3_12h = wr4_12h = ws4_12h = np.full(n, np.nan)
    
    # Align 12h indicators to 6h timeframe
    donch_high_6h = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_6h = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_6h[i]) or 
            np.isnan(donch_low_6h[i]) or
            np.isnan(ema_50_6h[i]) or
            np.isnan(wp_12h[i]) or
            np.isnan(wr3_12h[i]) or
            np.isnan(ws3_12h[i]) or
            np.isnan(wr4_12h[i]) or
            np.isnan(ws4_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike (avoid chop)
        vol_threshold = 1.8
        
        if position == 0:
            # Long: Price breaks above Donchian high AND above weekly R3 with volume
            # Only in uptrend (price above EMA50)
            if (close[i] > donch_high_6h[i] and 
                close[i] > wr3_12h[i] and 
                close[i] > ema_50_6h[i] and
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low AND below weekly S3 with volume
            # Only in downtrend (price below EMA50)
            elif (close[i] < donch_low_6h[i] and 
                  close[i] < ws3_12h[i] and 
                  close[i] < ema_50_6h[i] and
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below weekly S3 or Donchian low
            if close[i] < ws3_12h[i] or close[i] < donch_low_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above weekly R3 or Donchian high
            if close[i] > wr3_12h[i] or close[i] > donch_high_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0