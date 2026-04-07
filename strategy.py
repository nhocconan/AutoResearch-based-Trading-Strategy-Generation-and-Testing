#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian breakout with volume confirmation and weekly trend filter.
Uses weekly EMA200 for trend direction and daily Donchian channels for entry.
In bull markets (price > weekly EMA200): long on upper band breakout.
In bear markets (price < weekly EMA200): short on lower band breakout.
Volume must be above 20-period average to confirm breakout.
Designed for low trade frequency with clear trend-following logic.
Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
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
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)  # already shifted
    
    # === DAILY DONCHIAN CHANNELS (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(d_high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(d_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (use previous day's channels)
    high_max_aligned = align_htf_to_ltf(prices, df_1d, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1d, low_min)
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        bull_trend = close[i] > weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian band OR weekly trend turns bearish
            if close[i] < low_min_aligned[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian band OR weekly trend turns bullish
            if close[i] > high_max_aligned[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on weekly trend
            if bull_trend:
                # In bull market: long on upper band breakout
                if close[i] > high_max_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on lower band breakout
                if close[i] < low_min_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals