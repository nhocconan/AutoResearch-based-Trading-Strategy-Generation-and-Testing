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
    
    # Get weekly data for trend direction (1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate weekly EMA34 for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align weekly EMA to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for pivot points (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (Camarilla formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    s3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    r4_1d = close_1d + ((high_1d - low_1d) * 1.1)
    s4_1d = close_1d - ((high_1d - low_1d) * 1.1)
    
    # Align daily Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_weekly_ema = close[i] > ema_34_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: break above R4 with volume and uptrend
            if close[i] > r4_6h[i] and volume_filter and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: break below S4 with volume and downtrend
            elif close[i] < s4_6h[i] and volume_filter and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below R3
            if close[i] < r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above S3
            if close[i] > s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S4_Breakout_WeeklyEMA_Trend"
timeframe = "6h"
leverage = 1.0