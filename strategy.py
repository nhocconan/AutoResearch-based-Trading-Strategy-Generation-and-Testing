#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla H3/L3 breakout with 4h trend filter and volume confirmation
    # Uses 4h EMA50 for trend direction and 1d Camarilla levels (from prior 1d) for breakout
    # Volume > 1.5 * 20-period average confirms breakout strength
    # Discrete sizing 0.20 to minimize fee churn. Target: 20-40 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Camarilla H3/L3 levels (based on prior 1d bar's range)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        daily_range = high_1d[i-1] - low_1d[i-1]
        if daily_range > 0:
            camarilla_h3[i] = close_1d[i-1] + 1.1 * daily_range / 4
            camarilla_l3[i] = close_1d[i-1] - 1.1 * daily_range / 4
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend
        bullish_trend = close[i] > ema50_4h_aligned[i]
        bearish_trend = close[i] < ema50_4h_aligned[i]
        
        # Entry logic: Camarilla H3/L3 breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above H3 in bullish 4h trend
        if bullish_trend:
            long_entry = (close[i] > h3_aligned[i-1]) and volume_spike[i]
        # Short breakout: price breaks below L3 in bearish 4h trend
        elif bearish_trend:
            short_entry = (close[i] < l3_aligned[i-1]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < l3_aligned[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > h3_aligned[i]) or (not bullish_trend and not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_h3l3_breakout_trend_volume_v1"
timeframe = "1h"
leverage = 1.0