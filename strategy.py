#!/usr/bin/env python3
"""
12h_1w_1d_donchian_breakout_volume_v1
Strategy: 12h Donchian breakout with volume confirmation and 1w/1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses Donchian channel breakout (20-period high/low) on 12h timeframe for directional entries, confirmed by volume spike (>1.5x average volume), and filtered by 1w EMA40 and 1d EMA20 trend alignment. Designed to capture strong trending moves while avoiding false breakouts in choppy markets. Higher timeframes (1w/1d) provide trend direction, 12h provides entry timing. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_donchian_breakout_volume_v1"
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
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 12h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters: price above/both EMAs for long, below/both for short
        uptrend_1w = price_close > ema_40_1w_aligned[i]
        uptrend_1d = price_close > ema_20_1d_aligned[i]
        downtrend_1w = price_close < ema_40_1w_aligned[i]
        downtrend_1d = price_close < ema_20_1d_aligned[i]
        
        # Breakout conditions using Donchian channels
        breakout_up = price_close > high_max[i]
        breakout_down = price_close < low_min[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend (both 1w and 1d)
        long_signal = breakout_up and vol_confirmed and uptrend_1w and uptrend_1d
        
        # Short: downward breakout with volume in downtrend (both 1w and 1d)
        short_signal = breakout_down and vol_confirmed and downtrend_1w and downtrend_1d
        
        # Exit when price returns to the Donchian midpoint or opposite band
        midpoint = (high_max[i] + low_min[i]) / 2
        exit_long = position == 1 and price_close < midpoint
        exit_short = position == -1 and price_close > midpoint
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals