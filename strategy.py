#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use daily Donchian breakout (20-period) with daily trend filter (EMA20 > EMA50) and volume confirmation.
Breakouts above Donchian upper band with daily uptrend and high volume signal institutional buying.
Breakouts below Donchian lower band with daily downtrend and high volume signal institutional selling.
Exit on opposite band touch or trend reversal. Targets 12-37 trades/year to minimize fee drag.
Works in bull/bear via daily trend filter and volume confirmation for institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timezone = "12h"
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
    
    # Calculate daily EMA20 and EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily EMA to 12h timeframe
    ema20_12h = align_htf_to_ltf(prices, df_1d, daily_ema20)
    ema50_12h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Calculate daily Donchian channels (20-period)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Upper band: 20-period high
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation (24-period average on 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema20_12h[i]) or np.isnan(ema50_12h[i]) or 
            np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price touches Donchian lower band
            if close[i] <= donchian_low_12h[i]:
                exit_long = True
            # Exit if daily EMA20 crosses below EMA50 (trend reversal)
            elif ema20_12h[i] < ema50_12h[i] and ema20_12h[i-1] >= ema50_12h[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price touches Donchian upper band
            if close[i] >= donchian_high_12h[i]:
                exit_short = True
            # Exit if daily EMA20 crosses above EMA50 (trend reversal)
            elif ema20_12h[i] > ema50_12h[i] and ema20_12h[i-1] <= ema50_12h[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band with daily uptrend and volume confirmation
            long_entry = False
            if (close[i] > donchian_high_12h[i] and close[i-1] <= donchian_high_12h[i-1] and
                ema20_12h[i] > ema50_12h[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below Donchian lower band with daily downtrend and volume confirmation
            short_entry = False
            if (close[i] < donchian_low_12h[i] and close[i-1] >= donchian_low_12h[i-1] and
                ema20_12h[i] < ema50_12h[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals