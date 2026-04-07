#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v2
Hypothesis: Donchian breakout with volume confirmation and daily trend filter.
Breakouts above/below 20-period Donchian channels trigger entries only when
confirmed by volume surge and aligned with daily EMA50 trend. This structure
provides clean trend-following entries with built-in momentum, reducing false
breakouts. Works in bull markets via upward breakouts and in bear markets via
downward breakouts, with daily trend filter preventing counter-trend entries.
Targets 20-40 trades/year by requiring confluence of breakout, volume, and trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high/low
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align daily EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]
        breakout_down = close[i] < donchian_low[i-1]
        
        # Daily trend filter
        above_daily_ema50 = close[i] > ema50_1d_aligned[i]
        below_daily_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or below daily EMA50
            if close[i] < donchian_low[i] or below_daily_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or above daily EMA50
            if close[i] > donchian_high[i] or above_daily_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high with volume and above daily EMA50
            if breakout_up and vol_confirmed and above_daily_ema50:
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low with volume and below daily EMA50
            elif breakout_down and vol_confirmed and below_daily_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals