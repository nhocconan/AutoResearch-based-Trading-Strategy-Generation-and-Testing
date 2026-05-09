#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with daily trend filter and volume confirmation.
# Uses daily pivot levels for structure, daily EMA50 for trend filter, and volume spike for confirmation.
# Designed for low-frequency trading (12-37 trades/year) to minimize fee drag and improve generalization.
# Works in both bull and bear markets by requiring alignment with daily trend.
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily pivot points (using prior day's OHLC)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (daily_high + daily_low + daily_close) / 3.0
    # Resistance 1 = (2 * PP) - Low
    r1 = (2 * pp) - daily_low
    # Support 1 = (2 * PP) - High
    s1 = (2 * pp) - daily_high
    
    # Align daily pivot to 12h timeframe (with 1-bar delay for completed daily bar)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: spike above 2.0x 4-period average (2 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r1_12h[i]) or np.isnan(s1_12h[i])):
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
            # Long: price above daily S1, daily uptrend (price > EMA50), volume breakout
            if (close[i] > s1_12h[i] and 
                close[i] > ema_50_12h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below daily R1, daily downtrend (price < EMA50), volume breakdown
            elif (close[i] < r1_12h[i] and 
                  close[i] < ema_50_12h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily S1 or trend reversal
            if close[i] < s1_12h[i] or close[i] < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above daily R1 or trend reversal
            if close[i] > r1_12h[i] or close[i] > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals