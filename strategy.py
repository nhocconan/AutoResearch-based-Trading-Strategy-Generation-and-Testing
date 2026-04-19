#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Pivot_R1S1_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h pivot levels from previous 12h bar
    prev_close_12h = np.roll(close_12h, 1)
    prev_close_12h[0] = np.nan
    prev_high_12h = np.roll(high_12h, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h = np.roll(low_12h, 1)
    prev_low_12h[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_12h = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_12h = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 12.0
    
    # Align to 4h timeframe
    pivot_12h_4h = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_4h = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_4h = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: EMA34 on 12h (trend direction)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if np.isnan(pivot_12h_4h[i]) or np.isnan(r1_12h_4h[i]) or np.isnan(s1_12h_4h[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(ema_34_12h_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_34_12h_4h[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and above EMA34 (uptrend)
            if price > r1_12h_4h[i] and volume_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike and below EMA34 (downtrend)
            elif price < s1_12h_4h[i] and volume_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below S1 (reversal signal)
            if price < s1_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above R1 (reversal signal)
            if price > r1_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals