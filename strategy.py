#!/usr/bin/env python3
"""
Hypothesis: 12h EMA(34) trend filter with 1d RSI(14) mean reversion and volume spike.
- Long: EMA(34) rising, RSI < 30 (oversold), volume > 2x average
- Short: EMA(34) falling, RSI > 70 (overbought), volume > 2x average
- Exit: RSI crosses back to neutral (40-60 range) or EMA trend reversal
- Uses 1d RSI for mean reversion in ranging markets, EMA(34) for trend filter.
Designed for 12-37 trades/year (50-150 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    if len(gain) >= period:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    rsi = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    ema = np.full(len(close), np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    
    for i in range(period, len(close)):
        ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI (14-period) on 1d
    rsi_14_1d = calculate_rsi(close_1d, 14)
    
    # Calculate EMA (34-period) on 1d for trend filter
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Align to 12h timeframe
    rsi_14_1d_12h = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need RSI, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_14_1d_12h[i]) or np.isnan(ema_34_1d_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # EMA trend: rising if current > previous, falling if current < previous
        ema_rising = ema_34_1d_12h[i] > ema_34_1d_12h[i-1]
        ema_falling = ema_34_1d_12h[i] < ema_34_1d_12h[i-1]
        
        if position == 0:
            # Long: EMA rising, RSI < 30 (oversold), volume confirmation
            if ema_rising and rsi_14_1d_12h[i] < 30 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: EMA falling, RSI > 70 (overbought), volume confirmation
            elif ema_falling and rsi_14_1d_12h[i] > 70 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 40 or EMA trend turns down
            if rsi_14_1d_12h[i] >= 40 or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 60 or EMA trend turns up
            if rsi_14_1d_12h[i] <= 60 or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA34_RSI14_Volume"
timeframe = "12h"
leverage = 1.0