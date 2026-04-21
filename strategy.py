#!/usr/bin/env python3
"""
1d_1w_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Daily Donchian breakouts above/below 20-day high/low with volume confirmation and weekly trend filter (price above/below weekly 50 EMA). Works in bull markets via breakout continuation and in bear markets via mean reversion at band edges during low volatility. Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Donchian channels (20-period)
    upper_daily = np.full_like(high_daily, np.nan)
    lower_daily = np.full_like(low_daily, np.nan)
    
    for i in range(len(high_daily)):
        if i >= 19:
            upper_daily[i] = np.max(high_daily[i-19:i+1])
            lower_daily[i] = np.min(low_daily[i-19:i+1])
        elif i >= 0:
            upper_daily[i] = np.max(high_daily[:i+1])
            lower_daily[i] = np.min(low_daily[:i+1])
    
    # Load weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Calculate weekly EMA50
    alpha = 2.0 / (50 + 1)
    ema_weekly = np.full_like(close_weekly, np.nan)
    for i in range(len(close_weekly)):
        if i == 0:
            ema_weekly[i] = close_weekly[i]
        elif not np.isnan(ema_weekly[i-1]):
            ema_weekly[i] = alpha * close_weekly[i] + (1 - alpha) * ema_weekly[i-1]
    
    # Align daily Donchian and weekly EMA to 1d timeframe
    upper_daily_aligned = align_htf_to_ltf(prices, df_daily, upper_daily)
    lower_daily_aligned = align_htf_to_ltf(prices, df_daily, lower_daily)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-day average
    volume_avg = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        elif i > 0:
            volume_avg[i] = np.mean(volume[:i])
        else:
            volume_avg[i] = volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_daily_aligned[i]) or np.isnan(lower_daily_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = upper_daily_aligned[i]
        lower = lower_daily_aligned[i]
        ema_w = ema_weekly_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: breakout above upper band with volume and above weekly EMA
            if price > upper and vol_ok and price > ema_w:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower band with volume and below weekly EMA
            elif price < lower and vol_ok and price < ema_w:
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging markets: fade at bands when price near weekly EMA
            elif abs(price - ema_w) < (upper - lower) * 0.1:  # near weekly EMA
                if price > upper * 0.995 and vol_ok:  # near upper band
                    signals[i] = -0.25
                    position = -1
                elif price < lower * 1.005 and vol_ok:  # near lower band
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Long exit: price returns to lower band or breaks above upper band with weakening volume
            if price < lower or (price > upper and not vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to upper band or breaks below lower band with weakening volume
            if price > upper or (price < lower and not vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian20_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0