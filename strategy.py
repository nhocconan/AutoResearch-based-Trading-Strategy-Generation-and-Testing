#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h trend filter and 1d Elder Ray (Bull/Bear Power) for reversal signals.
# Uses 12h EMA50 for trend direction and 1d Bull/Bear Power to identify exhaustion in the trend.
# Designed for low trade frequency (12-25/year) to avoid fee drag in 6h timeframe.
# Works in both bull/bear markets by requiring alignment with 12h trend and Elder Ray divergence.
name = "6h_ElderRay_12hEMA50_1dTrend"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 13-period EMA for Elder Ray (standard period)
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = df_1d['high'].values - ema_13_1d  # High - EMA13
    bear_power_1d = df_1d['low'].values - ema_13_1d   # Low - EMA13
    
    # Align 1d indicators to 6h timeframe
    ema_50_6h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume filter: spike above 2.0x 12-period average (2 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 12)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_6h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: Bear power weakening (less negative) while price above 12h EMA50
            # Indicates selling pressure fading in uptrend
            if (bear_power_6h[i] > bear_power_6h[i-1] and  # Bear power improving (less negative)
                close[i] > ema_50_6h_aligned[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: Bull power weakening (less positive) while price below 12h EMA50
            # Indicates buying pressure fading in downtrend
            elif (bull_power_6h[i] < bull_power_6h[i-1] and  # Bull power deteriorating (less positive)
                  close[i] < ema_50_6h_aligned[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear power strengthening (more negative) or trend breakdown
            if bear_power_6h[i] < bear_power_6h[i-1] or close[i] < ema_50_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull power strengthening (more positive) or trend reversal
            if bull_power_6h[i] > bull_power_6h[i-1] or close[i] > ema_50_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals