#!/usr/bin/env python3
"""
1d_daily_donchian_20_breakout_1w_trend_volume_v1
Hypothesis: On 1d timeframe, use 20-day Donchian channel breakout with weekly trend filter (EMA20 > EMA50) and volume confirmation. Enter long when price breaks above upper band with bullish weekly trend and volume > 1.5x average; enter short when price breaks below lower band with bearish weekly trend and volume > 1.5x average. Exit on opposite band touch or trend reversal. Designed for low trade frequency (7-25/year) to minimize fee drag and work in both bull/bear markets via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_daily_donchian_20_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    ema20_weekly = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema20w_aligned = align_htf_to_ltf(prices, df_1w, ema20_weekly)
    ema50w_aligned = align_htf_to_ltf(prices, df_1w, ema50_weekly)
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after 50 for weekly EMA stability
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema20w_aligned[i]) or np.isnan(ema50w_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price touches or breaks below lower Donchian band
            if close[i] <= low_20[i]:
                exit_long = True
            # Exit if weekly EMA20 crosses below EMA50 (trend reversal)
            elif ema20w_aligned[i] < ema50w_aligned[i] and ema20w_aligned[i-1] >= ema50w_aligned[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price touches or breaks above upper Donchian band
            if close[i] >= high_20[i]:
                exit_short = True
            # Exit if weekly EMA20 crosses above EMA50 (trend reversal)
            elif ema20w_aligned[i] > ema50w_aligned[i] and ema20w_aligned[i-1] <= ema50w_aligned[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with bullish weekly trend and volume confirmation
            long_entry = False
            if (close[i] > high_20[i] and close[i-1] <= high_20[i-1] and
                ema20w_aligned[i] > ema50w_aligned[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below lower band with bearish weekly trend and volume confirmation
            short_entry = False
            if (close[i] < low_20[i] and close[i-1] >= low_20[i-1] and
                ema20w_aligned[i] < ema50w_aligned[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals