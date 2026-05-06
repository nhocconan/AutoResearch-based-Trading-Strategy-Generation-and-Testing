#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels with daily trend filter
# Long when price breaks above R4 (H4) with 1-day EMA34 uptrend and volume expansion
# Short when price breaks below S4 (L4) with 1-day EMA34 downtrend and volume expansion
# Uses weekly Camarilla levels for key support/resistance, daily trend for bias, volume for confirmation
# Designed to work in both bull and bear markets via breakout confirmation
# Target: 12-30 trades per year (50-120 over 4 years) with 0.25 position sizing

name = "6h_WeeklyCamarilla_R4L4_1dEMA34_VolumeExpansion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (using previous week's close)
    # H4 = Close + 1.1 * (High - Low)
    # L4 = Close - 1.1 * (High - Low)
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate R4 and S4 levels
    r4_level = weekly_close + 1.1 * (weekly_high - weekly_low)
    s4_level = weekly_close - 1.1 * (weekly_high - weekly_low)
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_level)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_level)
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1-day EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume expansion: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_expansion[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with daily uptrend and volume expansion
            if close[i] > r4_aligned[i] and close[i] > ema_34_aligned[i] and volume_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with daily downtrend and volume expansion
            elif close[i] < s4_aligned[i] and close[i] < ema_34_aligned[i] and volume_expansion[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S4 (support break)
            if close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R4 (resistance break)
            if close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals