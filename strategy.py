#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot levels from 1w + volume confirmation + chop regime filter
    # In trending regime (1w CHOP < 38.2): breakout long/short at Camarilla H3/L3 levels
    # In ranging regime (1w CHOP > 61.8): mean reversion at H4/L4 levels
    # Uses discrete sizing 0.25 to minimize fee churn. Target: 7-25 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for regime filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Choppiness Index (CHOP) - uses True Range and ATR
    def calculate_chop(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        atr = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing for ATR
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        chop = np.full(n, np.nan)
        for i in range(period, n):
            # Sum of true range over period
            tr_sum = np.sum(tr[i-period+1:i+1])
            # Highest high and lowest low over period
            hh = np.max(high[i-period+1:i+1])
            ll = np.min(low[i-period+1:i+1])
            if hh > ll and atr[i] > 0:
                chop[i] = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
        
        return chop
    
    chop_1w = calculate_chop(high_1w, low_1w, close_1w, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate 1w Camarilla pivot levels (based on previous week's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    # We need to shift by 1 week to avoid look-ahead (use previous week's data)
    high_1w_shifted = np.roll(high_1w, 1)  # Previous week's high
    low_1w_shifted = np.roll(low_1w, 1)    # Previous week's low
    close_1w_shifted = np.roll(close_1w, 1) # Previous week's close
    # Set first value to NaN (no previous week)
    high_1w_shifted[0] = np.nan
    low_1w_shifted[0] = np.nan
    close_1w_shifted[0] = np.nan
    
    # Calculate Camarilla levels based on previous week
    camarilla_h4 = close_1w_shifted + 1.5 * (high_1w_shifted - low_1w_shifted)
    camarilla_h3 = close_1w_shifted + 1.125 * (high_1w_shifted - low_1w_shifted)
    camarilla_l3 = close_1w_shifted - 1.125 * (high_1w_shifted - low_1w_shifted)
    camarilla_l4 = close_1w_shifted - 1.5 * (high_1w_shifted - low_1w_shifted)
    
    # Align Camarilla levels to 1d timeframe
    h4_1d = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    h3_1d = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_1d = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    l4_1d = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(chop_1w_aligned[i]) or np.isnan(h4_1d[i]) or np.isnan(h3_1d[i]) or 
            np.isnan(l3_1d[i]) or np.isnan(l4_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_1w_aligned[i]
        ranging = chop > 61.8
        trending = chop < 38.2
        
        long_entry = False
        short_entry = False
        
        if trending:
            # Trending regime: breakout at H3/L3 levels
            long_entry = (close[i] > h3_1d[i-1]) and volume_spike[i]
            short_entry = (close[i] < l3_1d[i-1]) and volume_spike[i]
        elif ranging:
            # Ranging regime: mean reversion at H4/L4 levels
            long_entry = (close[i] < l4_1d[i]) and volume_spike[i]  # Oversold bounce
            short_entry = (close[i] > h4_1d[i]) and volume_spike[i]  # Overbought rejection
        
        # Exit logic: opposite signal or regime change
        long_exit = (short_entry and position == 1) or (ranging and chop > 50 and position == 1)
        short_exit = (long_entry and position == -1) or (ranging and chop > 50 and position == -1)
        
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

name = "1d_1w_camarilla_chop_volume_v1"
timeframe = "1d"
leverage = 1.0