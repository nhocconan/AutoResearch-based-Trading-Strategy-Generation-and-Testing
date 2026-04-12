#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume confirmation
    # Uses 4h EMA50 for trend: only take breakouts in direction of 4h trend
    # Volume confirmation: volume > 2.0 * 20-period average to filter false breakouts
    # Session filter: 08-20 UTC to reduce noise trades
    # Discrete sizing 0.20 to minimize fee churn. Target: 15-35 trades/year per symbol.
    
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
    
    # Calculate daily Camarilla levels (based on previous day)
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6
    # L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12
    # L1 = close - 1.1*(high-low)/12
    
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_h2 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l2 = close_1d - 1.1 * (high_1d - low_1d) / 6
    camarilla_h1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_l1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Determine 4h trend
        bullish_trend = close[i] > ema50_4h_aligned[i]
        bearish_trend = close[i] < ema50_4h_aligned[i]
        
        # Entry logic: Camarilla H4 breakout long, L4 breakdown short
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Camarilla H4 in bullish trend
        if bullish_trend:
            long_entry = (close[i] > camarilla_h4_aligned[i]) and volume_spike[i]
        # Short breakdown: price breaks below Camarilla L4 in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l4_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_l4_aligned[i]) or \
                   (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > camarilla_h4_aligned[i]) or \
                    (not bullish_trend and not bearish_trend)
        
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

name = "1h_4d_camarilla_breakout_trend_volume_v1"
timeframe = "1h"
leverage = 1.0