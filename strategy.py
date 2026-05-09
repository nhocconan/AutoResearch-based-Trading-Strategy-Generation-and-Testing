# 12h_DailyPivot_Breakout_TrendFilter
# Strategy type: Daily pivot breakout with 1d trend filter and volume confirmation
# Rationale: Daily pivots provide key support/resistance levels for 12h timeframe.
# Uses 1d EMA for trend direction and volume spike for confirmation.
# Works in bull/bear by following 1d trend direction at daily support/resistance levels.
# Target: 15-35 trades/year per symbol with strict entry conditions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_DailyPivot_Breakout_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (daily_high + daily_low + daily_close) / 3.0
    # Resistance 1 = (2 * PP) - L
    r1 = (2 * pp) - daily_low
    # Support 1 = (2 * PP) - H
    s1 = (2 * pp) - daily_high
    # Resistance 2 = PP + (H - L)
    r2 = pp + (daily_high - daily_low)
    # Support 2 = PP - (H - L)
    s2 = pp - (daily_high - daily_low)
    
    # Align daily pivots to 12h timeframe (with 1-bar delay for completed daily bar)
    pp_12h = align_htf_to_ltf(prices, df_1d, pp)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: spike above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(pp_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24, 12h bars less sensitive)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Moderate session filter for 12h: avoid only the quietest hours (0-6 UTC)
        in_session = hour >= 6  # Start trading after 6 AM UTC
        
        if position == 0:
            # Long: price above daily S1, 1d uptrend (price > EMA34), volume breakout
            if (close[i] > s1_12h[i] and 
                close[i] > ema_34_12h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below daily R1, 1d downtrend (price < EMA34), volume breakdown
            elif (close[i] < r1_12h[i] and 
                  close[i] < ema_34_12h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily S2 or trend reversal
            if close[i] < s2_12h[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above daily R2 or trend reversal
            if close[i] > r2_12h[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals