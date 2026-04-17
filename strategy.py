#!/usr/bin/env python3
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
    
    # === 12h Donchian Channel (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper and lower bands
    upper_12h = np.full_like(high_12h, np.nan)
    lower_12h = np.full_like(low_12h, np.nan)
    period = 20
    for i in range(len(high_12h)):
        if i >= period - 1:
            upper_12h[i] = np.max(high_12h[i-period+1:i+1])
            lower_12h[i] = np.min(low_12h[i-period+1:i+1])
        elif i > 0:
            upper_12h[i] = np.max(high_12h[0:i+1])
            lower_12h[i] = np.min(low_12h[0:i+1])
        else:
            upper_12h[i] = high_12h[0]
            lower_12h[i] = low_12h[0]
    
    # Align to 12h timeframe (already at 12h resolution, but need proper alignment)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # === 12h Volume confirmation ===
    volume_12h = df_12h['volume'].values
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_12h[0:i+1])
        else:
            vol_ma_20[i] = volume_12h[0]
    
    vol_confirm = volume_12h > vol_ma_20 * 1.5
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm)
    
    # === 1d ATR for stop loss and position sizing ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate ATR (14-period)
    atr_1d = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            if i == 0:
                atr_1d[i] = tr[i]
            else:
                atr_1d[i] = np.mean(tr[0:i+1])
        else:
            atr_1d[i] = np.mean(tr[i-13:i+1])
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1w Trend filter (EMA 50) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA 50
    ema_50 = np.full_like(close_1w, np.nan)
    multiplier = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_50[i] = close_1w[i]
        elif np.isnan(ema_50[i-1]):
            ema_50[i] = close_1w[i]
        else:
            ema_50[i] = (close_1w[i] - ema_50[i-1]) * multiplier + ema_50[i-1]
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + above weekly EMA
            if (close[i] > upper_12h_aligned[i] and 
                vol_confirm_aligned[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian + volume confirmation + below weekly EMA
            elif (close[i] < lower_12h_aligned[i] and 
                  vol_confirm_aligned[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price closes below lower Donchian OR ATR-based stop
            if (close[i] < lower_12h_aligned[i] or 
                close[i] < high[i] - 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above upper Donchian OR ATR-based stop
            if (close[i] > upper_12h_aligned[i] or 
                close[i] > low[i] + 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_VolumeWeeklyTrend_ATRStop"
timeframe = "12h"
leverage = 1.0