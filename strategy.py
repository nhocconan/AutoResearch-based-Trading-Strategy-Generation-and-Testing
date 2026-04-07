#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
In bull market (1d close > 1d EMA50): long on break above 4h Donchian upper band.
In bear market (1d close < 1d EMA50): short on break below 4h Donchian lower band.
Volume must be above 20-period average to confirm breakout.
Target: 15-30 trades per year per symbol (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v3"
timeframe = "4h"
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
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)  # already shifted
    
    # === 4H DONCHIAN CHANNELS (20-period) ===
    lookback = 20
    # Calculate highest high and lowest low over lookback period
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band OR trend turns bearish
            if close[i] < lowest_low[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band OR trend turns bullish
            if close[i] > highest_high[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 1d trend and Donchian breakout
            if bull_trend:
                # In bull market: long on break above Donchian upper band
                if high[i] > highest_high[i] and close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on break below Donchian lower band
                if low[i] < lowest_low[i] and close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals