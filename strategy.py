#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Trend
Hypothesis: Combines 12-hour Camarilla pivot levels with 1-day trend filter (EMA50) and volume confirmation.
Trades breakouts of key pivot levels (H3/L3) in the direction of higher timeframe trend to avoid counter-trend whipsaws.
Designed for 12-37 trades/year per symbol with high win rate during trends.
Works in bull/bear by following 1d trend direction - avoids counter-trend losses.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range = high - low
    daily_range = df_1d['high'] - df_1d['low']
    # Camarilla levels
    # H4 = close + 1.5 * (high - low) * 1.1
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low) * 1.1
    camarilla_h3 = close_1d + 1.1 * daily_range
    camarilla_l3 = close_1d - 1.1 * daily_range
    
    # Align pivot levels to 12h timeframe (previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_h3 = close[i] > camarilla_h3_aligned[i-1]  # Break above H3
        breakdown_l3 = close[i] < camarilla_l3_aligned[i-1]  # Break below L3
        
        # Entry conditions: only trade in direction of 1d trend
        long_entry = breakout_h3 and volume_filter and uptrend
        short_entry = breakdown_l3 and volume_filter and downtrend
        
        # Exit conditions: return to opposite Camarilla level or trend reversal
        long_exit = (close[i] < camarilla_l3_aligned[i]) or (not uptrend)  # Break below L3 or trend change
        short_exit = (close[i] > camarilla_h3_aligned[i]) or (not downtrend)  # Break above H3 or trend change
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals