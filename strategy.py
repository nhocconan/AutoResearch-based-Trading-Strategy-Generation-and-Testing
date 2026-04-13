#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume confirmation
    # Uses 12h EMA50 for trend direction (HTF) to avoid counter-trend trades
    # Camarilla H3/L3 levels from 1d for precise entry/exit levels
    # Volume > 1.5x 20-period average confirms breakout strength
    # Target: 20-40 trades/year (80-160 total over 4 years) for low fee drag
    # Works in bull via long bias, in bear via short bias from 12h EMA50 filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_12h[49] = np.mean(close_12h[:50])  # SMA50 as seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous 1d bar's OHLC (aligned to current 4h bar)
        prev_high = df_1d['high'].iloc[i-1] if i-1 < len(df_1d) else df_1d['high'].iloc[-1]
        prev_low = df_1d['low'].iloc[i-1] if i-1 < len(df_1d) else df_1d['low'].iloc[-1]
        prev_close = df_1d['close'].iloc[i-1] if i-1 < len(df_1d) else df_1d['close'].iloc[-1]
        
        range_val = prev_high - prev_low
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align 12h EMA50 to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at Camarilla levels
        long_breakout = close[i] > camarilla_h3[i]
        short_breakout = close[i] < camarilla_l3[i]
        
        # Trend filter from 12h EMA50
        bullish_trend = close[i] > ema_12h_aligned[i]
        bearish_trend = close[i] < ema_12h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike[i]
        short_entry = short_breakout and bearish_trend and volume_spike[i]
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = short_breakout or (close[i] < ema_12h_aligned[i])
        short_exit = long_breakout or (close[i] > ema_12h_aligned[i])
        
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

name = "4h_1d_camarilla_h3l3_12h_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0