#!/usr/bin/env python3
"""
daily_donchian_20_breakout_1w_trend_volume_v1
Hypothesis: On daily timeframe, breakout above/below 20-day Donchian channel with weekly trend filter (EMA20 > EMA50) and volume confirmation (volume > 1.5x 20-day average). Enter long on upper band breakout, short on lower band breakout. Exit on opposite band touch or trend reversal. This captures medium-term momentum while avoiding false breakouts in low-volume, ranging markets. Works in bull via breakout continuation and in bear via short breakdowns. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_donchian_20_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA20 > EMA50 for uptrend
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Weekly timeframe data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 and EMA50 for stronger trend filter
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema20[i]) or np.isnan(ema50[i]) or
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(weekly_ema50_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: both daily and weekly EMAs agree
        daily_uptrend = ema20[i] > ema50[i]
        weekly_uptrend = weekly_ema20_aligned[i] > weekly_ema50_aligned[i]
        daily_downtrend = ema20[i] < ema50[i]
        weekly_downtrend = weekly_ema20_aligned[i] < weekly_ema50_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price touches lower Donchian band (contrarian exit)
            if close[i] <= low_20[i]:
                exit_long = True
            # Exit if trend turns down on both timeframes
            elif not daily_uptrend and not weekly_uptrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price touches upper Donchian band (contrarian exit)
            if close[i] >= high_20[i]:
                exit_short = True
            # Exit if trend turns up on both timeframes
            elif not daily_downtrend and not weekly_downtrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band with uptrend and volume
            long_entry = False
            if (close[i] > high_20[i] and close[i-1] <= high_20[i-1] and
                daily_uptrend and weekly_uptrend and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below lower Donchian band with downtrend and volume
            short_entry = False
            if (close[i] < low_20[i] and close[i-1] >= low_20[i-1] and
                daily_downtrend and weekly_downtrend and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals