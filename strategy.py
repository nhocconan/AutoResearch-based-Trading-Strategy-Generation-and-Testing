#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter + volume spike confirmation
    # Long: price > H3 + price > 1d EMA50 + volume > 2.0x 20-period average
    # Short: price < L3 + price < 1d EMA50 + volume > 2.0x 20-period average
    # Exit: opposite Camarilla breakout OR price crosses 1d EMA50
    # Camarilla levels from prior 1d provide strong support/resistance
    # 1d EMA50 trend filter reduces whipsaw, volume confirmation ensures momentum
    # Discrete position sizing 0.25 minimizes fee churn while maintaining edge
    # Target: 12-37 trades/year on 12h timeframe (low fee drag, high win rate)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 with min_periods
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])  # SMA50 as seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate Camarilla levels from prior 1d bar (H3, L3, H4, L4)
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_high = np.full(len(close_1d), np.nan)
    camarilla_low = np.full(len(close_1d), np.nan)
    camarilla_high_strong = np.full(len(close_1d), np.nan)  # H4
    camarilla_low_strong = np.full(len(close_1d), np.nan)   # L4
    
    for i in range(1, len(close_1d)):
        # Use prior completed 1d bar (i-1) to avoid look-ahead
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        rang = prev_high - prev_low
        
        camarilla_high[i] = prev_close + 1.1 * rang / 4  # H3
        camarilla_low[i] = prev_close - 1.1 * rang / 4   # L3
        camarilla_high_strong[i] = prev_close + 1.1 * rang / 2  # H4
        camarilla_low_strong[i] = prev_close - 1.1 * rang / 2   # L4
    
    # Align 1d indicators to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_high_strong_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high_strong)
    camarilla_low_strong_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low_strong)
    
    # Volume confirmation: >2.0x 20-period average (tight filter)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        long_breakout = close[i] > camarilla_high_aligned[i]
        short_breakout = close[i] < camarilla_low_aligned[i]
        
        # Trend filter from 1d EMA50
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike[i]
        short_entry = short_breakout and bearish_trend and volume_spike[i]
        
        # Exit logic: opposite breakout or trend reversal
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

name = "12h_1d_camarilla_h3l3_ema50_volume_v1"
timeframe = "12h"
leverage = 1.0