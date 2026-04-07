#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout with daily trend filter and volume confirmation.
Uses daily EMA50 for trend direction and Donchian(20) channels for breakout entries.
In uptrend (price > daily EMA50): long on upper band breakout.
In downtrend (price < daily EMA50): short on lower band breakout.
Volume must be above 20-period average to confirm breakout.
Exit on opposite band touch or trend reversal.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_trend_volume_v1"
timeframe = "12h"
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
    
    # === DAILY TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)  # already shifted
    
    # === DONCHIAN CHANNELS (LTF) ===
    # Use 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(daily_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA
        uptrend = close[i] > daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian band OR trend turns down
            if close[i] <= donchian_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian band OR trend turns up
            if close[i] >= donchian_high[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on daily trend
            if uptrend:
                # In uptrend: long on upper band breakout
                if close[i] >= donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In downtrend: short on lower band breakout
                if close[i] <= donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals