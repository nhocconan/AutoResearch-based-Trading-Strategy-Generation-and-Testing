#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter + volume spike
    # Long: price > H3 + price > 4h EMA50 + volume > 2.0x 20-period average
    # Short: price < L3 + price < 4h EMA50 + volume > 2.0x 20-period average
    # Exit: opposite Camarilla breakout OR price crosses 4h EMA50
    # Session filter: 08-20 UTC to avoid low-volume Asian session noise
    # Discrete position size: 0.20 (20%) to control drawdown and minimize fee churn
    # Target: 15-37 trades/year (60-150 over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA50 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA50 with min_periods
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])  # SMA50 as seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    
    # Calculate 4h Camarilla levels (H3, L3) based on previous day's range
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = np.full(len(close_4h), np.nan)
    camarilla_l3 = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Use previous 4h bar's range for today's levels
        prev_high = high_4h[i-1]
        prev_low = low_4h[i-1]
        prev_close = close_4h[i-1]
        camarilla_h3[i] = prev_close + 1.1 * (prev_high - prev_low) / 2
        camarilla_l3[i] = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align 4h indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside trading session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Trend filter from 4h EMA50
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike[i]
        short_entry = short_breakout and bearish_trend and volume_spike[i]
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = short_breakout or (close[i] < ema_4h_aligned[i])
        short_exit = long_breakout or (close[i] > ema_4h_aligned[i])
        
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

name = "1h_4h_camarilla_breakout_ema50_volume_v1"
timeframe = "1h"
leverage = 1.0