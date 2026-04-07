#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use weekly Donchian channels (20-period) for breakout signals with 1w EMA trend filter and volume confirmation. Enter long when price breaks above upper band with weekly EMA20 > EMA50 and volume > 1.5x average; enter short when price breaks below lower band with weekly EMA20 < EMA50 and volume > 1.5x average. Exit when price reaches opposite Donchian band or EMA crossover reverses. This strategy captures momentum from institutional breakouts while using weekly trend filter to avoid counter-trend trades. Volume confirmation ensures breakouts have participation. Designed for 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA20 and EMA50 for trend filter
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate weekly Donchian channels (20-period) from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels: highest high and lowest low of past 20 weekly bars
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian channels
    upper_band = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (shifted by 1 week for look-ahead prevention)
    upper_band_12h = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_12h = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Volume confirmation (24-period average on 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(upper_band_12h[i]) or np.isnan(lower_band_12h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches lower band (opposite side)
            if close[i] <= lower_band_12h[i]:
                exit_long = True
            # Exit if EMA20 crosses below EMA50 (trend reversal)
            elif ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price reaches upper band (opposite side)
            if close[i] >= upper_band_12h[i]:
                exit_short = True
            # Exit if EMA20 crosses above EMA50 (trend reversal)
            elif ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with EMA20 > EMA50 and volume confirmation
            long_entry = False
            if (close[i] > upper_band_12h[i] and close[i-1] <= upper_band_12h[i-1] and
                ema20[i] > ema50[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below lower band with EMA20 < EMA50 and volume confirmation
            short_entry = False
            if (close[i] < lower_band_12h[i] and close[i-1] >= lower_band_12h[i-1] and
                ema20[i] < ema50[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals