#!/usr/bin/env python3
# 1d_1wDonchian_20_1wEMA34_Trend_Volume
# Uses weekly Donchian channel (20-period) breakout with weekly EMA34 trend filter and weekly volume confirmation.
# Designed for 1d timeframe to capture major trend continuations with proper risk management.
# Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing.
# Works in bull markets by following upward breaks and in bear markets by following downward breaks.

name = "1d_1wDonchian_20_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for Donchian, EMA, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Upper band: highest high of last 20 weeks
    upper = np.full_like(high_1w, np.nan)
    for i in range(20, len(high_1w)):
        upper[i] = np.max(high_1w[i-20:i])
    
    # Lower band: lowest low of last 20 weeks
    lower = np.full_like(low_1w, np.nan)
    for i in range(20, len(low_1w)):
        lower[i] = np.min(low_1w[i-20:i])
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1w > (1.5 * vol_ma_20)  # Moderate volume confirmation
    
    # Align weekly indicators to daily timeframe
    upper_daily = align_htf_to_ltf(prices, df_1w, upper)
    lower_daily = align_htf_to_ltf(prices, df_1w, lower)
    ema_34_daily = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    volume_spike_daily = align_htf_to_ltf(prices, df_1w, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_daily[i]) or np.isnan(lower_daily[i]) or 
            np.isnan(ema_34_daily[i]) or np.isnan(volume_spike_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: break above upper Donchian with uptrend and volume
            if close[i] > upper_daily[i] and close[i] > ema_34_daily[i] and volume_spike_daily[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below lower Donchian with downtrend and volume
            elif close[i] < lower_daily[i] and close[i] < ema_34_daily[i] and volume_spike_daily[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns below EMA34 or breaks below lower Donchian
            # Minimum holding period of 5 days to reduce churn
            if bars_since_entry >= 5 and (close[i] < ema_34_daily[i] or close[i] < lower_daily[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above EMA34 or breaks above upper Donchian
            # Minimum holding period of 5 days to reduce churn
            if bars_since_entry >= 5 and (close[i] > ema_34_daily[i] or close[i] > upper_daily[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals