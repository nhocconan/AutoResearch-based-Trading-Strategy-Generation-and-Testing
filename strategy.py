#!/usr/bin/env python3
"""
1d_1w_VolatilityBreakout_Momentum
Hypothesis: Use weekly volatility breakout with daily momentum confirmation.
Long when price breaks above weekly high with daily RSI > 50 and volume > 1.5x average.
Short when price breaks below weekly low with daily RSI < 50 and volume > 1.5x average.
Exit when price crosses back through weekly midpoint.
Designed for 1d timeframe to capture weekly momentum with ~10-25 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume and RSI filters prevent false breakouts in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for volatility breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly high/low/midpoint (based on previous week)
    weekly_high = np.full_like(close_1w, np.nan)
    weekly_low = np.full_like(close_1w, np.nan)
    weekly_mid = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(high_1w)):
        weekly_high[i] = high_1w[i-1]
        weekly_low[i] = low_1w[i-1]
        weekly_mid[i] = (high_1w[i-1] + low_1w[i-1]) / 2.0
    
    # Shift to align with current week (levels are based on previous week)
    weekly_high = np.roll(weekly_high, 1)
    weekly_low = np.roll(weekly_low, 1)
    weekly_mid = np.roll(weekly_mid, 1)
    weekly_high[0] = np.nan
    weekly_low[0] = np.nan
    weekly_mid[0] = np.nan
    
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Daily indicators for confirmation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average (20-period)
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if weekly indicators not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_mid_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            volume_ok = vol > 1.5 * vol_ma[i]
        else:
            volume_ok = False
        
        # RSI filter
        rsi_ok_long = rsi[i] > 50
        rsi_ok_short = rsi[i] < 50
        
        if position == 0:
            # Long conditions: break above weekly high + volume + RSI confirmation
            if price > weekly_high_aligned[i] and volume_ok and rsi_ok_long:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below weekly low + volume + RSI confirmation
            elif price < weekly_low_aligned[i] and volume_ok and rsi_ok_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below weekly midpoint
            if price < weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly midpoint
            if price > weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_VolatilityBreakout_Momentum"
timeframe = "1d"
leverage = 1.0