#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume confirmation
    # Uses 12h timeframe for lower trade frequency (target: 12-37/year) to minimize fee drag
    # Breakout at Camarilla H4/L4 levels (stronger than H3/L3) with 1d EMA50 trend filter
    # Volume spike (>2.0x 24-period average) confirms institutional participation
    # Designed to work in both bull and bear markets via trend-adaptive breakout direction
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 12h data for Camarilla pivot levels (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Camarilla pivot levels (H4/L4 for stronger breakouts)
    camarilla_h4_12h = np.full(len(df_12h), np.nan)
    camarilla_l4_12h = np.full(len(df_12h), np.nan)
    pivot_12h = np.full(len(df_12h), np.nan)
    
    for i in range(1, len(df_12h)):
        high_val = high_12h[i-1]
        low_val = low_12h[i-1]
        close_val = close_12h[i-1]
        pivot_val = (high_val + low_val + close_val) / 3.0
        range_val = high_val - low_val
        
        pivot_12h[i] = pivot_val
        camarilla_h4_12h[i] = pivot_val + range_val * 1.1 / 2.0  # H4
        camarilla_l4_12h[i] = pivot_val - range_val * 1.1 / 2.0  # L4
    
    # Align 12h Camarilla levels to 12h timeframe (no alignment needed as primary TF)
    camarilla_h4_12h_aligned = camarilla_h4_12h
    camarilla_l4_12h_aligned = camarilla_l4_12h
    
    # Volume confirmation: volume > 2.0 * 24-period average (12h)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_h4_12h_aligned[i]) or 
            np.isnan(camarilla_l4_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Camarilla H4/L4 breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above camarilla H4 in bullish trend with volume
        if bullish_trend:
            long_entry = (close[i] > camarilla_h4_12h_aligned[i]) and volume_spike[i]
        # Short breakout: price breaks below camarilla L4 in bearish trend with volume
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l4_12h_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite camarilla level or trend reversal
        long_exit = bearish_trend and close[i] < camarilla_l4_12h_aligned[i]
        short_exit = bullish_trend and close[i] > camarilla_h4_12h_aligned[i]
        
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

name = "12h_1d_camarilla_h4l4_ema50_volume_v1"
timeframe = "12h"
leverage = 1.0