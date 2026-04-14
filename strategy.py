#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Pivot Point (PP) breakout with volume confirmation and weekly trend filter
# Daily Pivot Points provide key support/resistance levels derived from prior day's price action
# Breakouts above R1 or below S1 with volume confirmation indicate institutional participation
# Weekly trend filter (price above/below weekly EMA20) ensures we trade with the higher timeframe trend
# Target: 15-25 trades/year per symbol to avoid fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter
    ema_len = 20
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily pivot points from previous day's OHLC
    # Pivot Point (PP) = (High + Low + Close) / 3
    # Resistance 1 (R1) = (2 * PP) - Low
    # Support 1 (S1) = (2 * PP) - High
    pp = (high + low + close) / 3.0
    r1 = (2 * pp) - low
    s1 = (2 * pp) - high
    
    # Shift pivot levels by 1 day to avoid look-ahead (today's levels based on yesterday's data)
    pp_shifted = np.roll(pp, 1)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    pp_shifted[0] = np.nan
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(pp_shifted[i]) or 
            np.isnan(r1_shifted[i]) or
            np.isnan(s1_shifted[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + volume + uptrend
            if (close[i] > r1_shifted[i] and 
                volume_confirmed and 
                uptrend):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below S1 + volume + downtrend
            elif (close[i] < s1_shifted[i] and 
                  volume_confirmed and 
                  downtrend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot point (mean reversion to mean)
            if close[i] < pp_shifted[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot point (mean reversion to mean)
            if close[i] > pp_shifted[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_DailyPivot_Breakout_Volume_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0