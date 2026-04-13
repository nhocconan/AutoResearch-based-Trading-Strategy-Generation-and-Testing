#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot (H3/L3) breakout with 1d trend filter (EMA50) and volume confirmation
    # Uses 1d EMA50 for trend direction (HTF) to avoid counter-trend trades
    # Camarilla H3/L3 breakout on 12h for entry timing (mean reversion in range, breakout in trend)
    # Volume > 1.5x 20-period average confirms breakout strength
    # Target: 12-30 trades/year (50-120 total over 4 years) for low fee drag
    # Works in bull via long bias from 1d EMA50, in bear via short bias from same filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])  # SMA50 as seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate Camarilla pivot levels (H3, L3) from 1d OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        camarilla_h3[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i]) / 4
        camarilla_l3[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i]) / 4
    
    # Get 12h Donchian(20) for breakout confirmation (optional structure)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align 1d indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA50
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Entry logic: Camarilla breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike[i]
        short_entry = short_breakout and bearish_trend and volume_spike[i]
        
        # Exit logic: opposite Camarilla breakout or trend reversal
        long_exit = short_breakout or (close[i] < ema_1d_aligned[i])
        short_exit = long_breakout or (close[i] > ema_1d_aligned[i])
        
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

name = "12h_1d_camarilla_breakout_ema50_volume_v1"
timeframe = "12h"
leverage = 1.0