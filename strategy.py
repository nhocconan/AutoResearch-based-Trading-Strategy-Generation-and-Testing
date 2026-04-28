#!/usr/bin/env python3
# Hypothesis: 12h Camarilla pivot breakout with daily EMA34 trend filter and volume spike confirmation.
# Camarilla levels (R1, S1) act as intraday support/resistance. Breakouts above R1 or below S1
# with volume > 2x average indicate institutional participation. Daily EMA34 filters trend direction.
# Works in bull/bear: long only when price > daily EMA34, short only when price < daily EMA34.
# Target: 15-35 trades/year on 12h timeframe (60-140 total over 4 years).

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
    
    # Get daily data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from previous day
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R1 and S1 for breakout entries
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First period
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculation
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 2x 24-period average (2 days of 12h bars)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_aligned[i]
        
        # Trend filter: price above/below daily EMA34
        uptrend = price > ema_trend
        downtrend = price < ema_trend
        
        # Breakout conditions with volume confirmation
        long_breakout = price > r1_aligned[i] and volume_filter[i]
        short_breakout = price < s1_aligned[i] and volume_filter[i]
        
        # Exit conditions: return to EMA34 or opposite Camarilla level
        long_exit = (price <= ema_trend) or (price < s1_aligned[i])
        short_exit = (price >= ema_trend) or (price > r1_aligned[i])
        
        # Handle entries and exits
        if long_breakout and uptrend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and downtrend and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0