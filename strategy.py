#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Pivot_R4S4_Breakout_Volume_Spike_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels from previous weekly bar
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w = np.roll(high_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w = np.roll(low_1w, 1)
    prev_low_1w[0] = np.nan
    
    # Weekly pivot = (H + L + C) / 3
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    # R4 = C + (H - L) * 1.1 / 2
    r4_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 2.0
    
    # Align to 6h timeframe
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_1w_6h = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_6h = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_6h[i]) or np.isnan(r4_1w_6h[i]) or np.isnan(s4_1w_6h[i]) or \
           np.isnan(ema_34_1d_6h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.5x average
        volume_spike = vol > 2.5 * vol_ma
        
        # Trend filter: price above/below daily EMA34
        trend_up = price > ema_34_1d_6h[i]
        trend_down = price < ema_34_1d_6h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R4 with volume spike and uptrend
            if price > r4_1w_6h[i] and volume_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S4 with volume spike and downtrend
            elif price < s4_1w_6h[i] and volume_spike and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly pivot (reversal signal)
            if price < pivot_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly pivot (reversal signal)
            if price > pivot_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals