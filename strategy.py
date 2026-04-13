#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1w Supertrend trend filter and price action at 1d key levels.
# Long: Price > 1d pivot point + 1w Supertrend bullish + volume > 1.5x average volume (20-period).
# Short: Price < 1d pivot point + 1w Supertrend bearish + volume > 1.5x average volume.
# Uses 1w Supertrend for trend direction (avoids counter-trend trades), 1d pivot points for mean reversion zones, and volume for confirmation.
# Timeframe: 12h balances trade frequency and signal quality. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Supertrend (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr0 = high_1w - low_1w
    tr1 = np.abs(high_1w - np.roll(close_1w, 1))
    tr2 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    
    # ATR
    atr = np.full_like(tr, np.nan)
    for i in range(atr_period, len(tr)):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Supertrend calculation
    hl_avg = (high_1w + low_1w) / 2
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr
    
    supertrend = np.full_like(close_1w, np.nan)
    trend = np.full_like(close_1w, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            trend[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1]) if not np.isnan(supertrend[i-1]) else lower_band[i]
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1]) if not np.isnan(supertrend[i-1]) else upper_band[i]
    
    # 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points (using previous day's data)
    pivot = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        pivot[i] = (ph + pl + pc) / 3
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1w Supertrend to 12h
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    # Align 1d pivot to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        st = supertrend_aligned[i]
        tr = trend_aligned[i]
        piv = pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price > pivot + uptrend + volume confirmation
            if (price > piv and tr == 1 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price < pivot + downtrend + volume confirmation
            elif (price < piv and tr == -1 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below pivot OR trend turns bearish
            if price < piv or tr == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above pivot OR trend turns bullish
            if price > piv or tr == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Supertrend_1d_Pivot_Volume"
timeframe = "12h"
leverage = 1.0