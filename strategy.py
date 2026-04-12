#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla breakout with 12h trend filter + volume spike
    # Uses 12h EMA34 trend filter: only take breakouts in direction of 12h trend
    # Volume confirmation: volume > 2.5 * 20-period average to filter false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Camarilla levels H3/L3 (most significant levels)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        daily_range = high_1d[i-1] - low_1d[i-1]
        if daily_range > 0:
            camarilla_h3[i] = close_1d[i-1] + 1.1 * daily_range / 4
            camarilla_l3[i] = close_1d[i-1] - 1.1 * daily_range / 4
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend
        bullish_trend = close[i] > ema34_12h_aligned[i]
        bearish_trend = close[i] < ema34_12h_aligned[i]
        
        # Entry logic: Camarilla H3/L3 breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above H3 in bullish 12h trend
        if bullish_trend:
            long_entry = (close[i] > h3_aligned[i-1]) and volume_spike[i]
        # Short breakout: price breaks below L3 in bearish 12h trend
        elif bearish_trend:
            short_entry = (close[i] < l3_aligned[i-1]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < l3_aligned[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > h3_aligned[i]) or (not bullish_trend and not bearish_trend)
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_camarilla_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0