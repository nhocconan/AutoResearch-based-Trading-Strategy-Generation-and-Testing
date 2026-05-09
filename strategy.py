# 6h_WeeklyPivot_TrendFollowing_VolumeFilter
# Strategy type: Weekly pivot levels with 1d trend filter and volume confirmation
# Rationale: Weekly pivots capture long-term structure; 1d trend filters direction; volume confirms breakouts.
# Works in bull/bear by following 1d trend direction at weekly support/resistance levels.
# Volume filter ensures momentum; 6h timeframe balances trade frequency and signal quality.
# Target: 20-40 trades/year per symbol with strict entry conditions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_TrendFollowing_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Resistance 1 = (2 * PP) - L
    r1 = (2 * pp) - weekly_low
    # Support 1 = (2 * PP) - H
    s1 = (2 * pp) - weekly_high
    # Resistance 2 = PP + (H - L)
    r2 = pp + (weekly_high - weekly_low)
    # Support 2 = PP - (H - L)
    s2 = pp - (weekly_high - weekly_low)
    # Resistance 3 = H + 2*(PP - L)
    r3 = weekly_high + 2 * (pp - weekly_low)
    # Support 3 = L - 2*(H - PP)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivots to 6h timeframe (with 1-bar delay for completed weekly bar)
    pp_6h = align_htf_to_ltf(prices, df_1w, pp)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: spike above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_6h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24, 6h bars less sensitive)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Less strict session for 6h: avoid only the quietest hours (0-6 UTC)
        in_session = hour >= 6  # Start trading after 6 AM UTC
        
        if position == 0:
            # Long: price above weekly S1, 1d uptrend (price > EMA34), volume breakout
            if (close[i] > s1_6h[i] and 
                close[i] > ema_34_6h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly R1, 1d downtrend (price < EMA34), volume breakdown
            elif (close[i] < r1_6h[i] and 
                  close[i] < ema_34_6h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly S2 or trend reversal
            if close[i] < s2_6h[i] or close[i] < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly R2 or trend reversal
            if close[i] > r2_6h[i] or close[i] > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals