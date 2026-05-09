#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points (standard formula)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (daily_high + daily_low + daily_close) / 3.0
    # Range = H - L
    rng = daily_high - daily_low
    # Resistance 1 = C + (H - L) * 1.1 / 12
    r1 = daily_close + rng * 1.1 / 12
    # Support 1 = C - (H - L) * 1.1 / 12
    s1 = daily_close - rng * 1.1 / 12
    # Resistance 2 = C + (H - L) * 1.1 / 6
    r2 = daily_close + rng * 1.1 / 6
    # Support 2 = C - (H - L) * 1.1 / 6
    s2 = daily_close - rng * 1.1 / 6
    
    # Align daily Camarilla to 1h timeframe (with 1-bar delay for completed daily bar)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    r2_1h = align_htf_to_ltf(prices, df_1d, r2)
    s2_1h = align_htf_to_ltf(prices, df_1d, s2)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: spike above 2.0x 24-period average (1 day of 1h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(r2_1h[i]) or np.isnan(s2_1h[i])):
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
            # Long: price above daily S1, 4h uptrend (price > EMA50), volume breakout
            if (close[i] > s1_1h[i] and 
                close[i] > ema_50_1h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price below daily R1, 4h downtrend (price < EMA50), volume breakdown
            elif (close[i] < r1_1h[i] and 
                  close[i] < ema_50_1h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price below daily S2 or trend reversal
            if close[i] < s2_1h[i] or close[i] < ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price above daily R2 or trend reversal
            if close[i] > r2_1h[i] or close[i] > ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals