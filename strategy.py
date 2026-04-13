#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# Long: price touches Camarilla L3 support + price above 1d EMA50 + volume > 1.5x avg volume
# Short: price touches Camarilla H3 resistance + price below 1d EMA50 + volume > 1.5x avg volume
# Camarilla levels calculated from 1d data: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
# Trend filter: only take longs when price > EMA50, shorts when price < EMA50
# Volume confirmation reduces false reversals
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in both bull and bear markets by using 1d EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3 and L3)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
        else:
            rng = high_1d[i-1] - low_1d[i-1]
            camarilla_h3[i] = close_1d[i-1] + 1.1 * rng / 6
            camarilla_l3[i] = close_1d[i-1] - 1.1 * rng / 6
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (24-period = 24*4h = 6 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    # Align 1d data to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price touches L3 support + above EMA50 + volume confirmation
            if (price <= l3 and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price touches H3 resistance + below EMA50 + volume confirmation
            elif (price >= h3 and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above EMA50 or touches H3 resistance
            if (price >= ema_trend or
                price >= h3):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below EMA50 or touches L3 support
            if (price <= ema_trend or
                price <= l3):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Reversal"
timeframe = "4h"
leverage = 1.0