#!/usr/bin/env python3
"""
Hypothesis: 1h momentum with 4h EMA trend filter and 1d volume confirmation.
Uses 4h EMA (50) for trend direction, 1h RSI (14) for momentum entry, and 1d volume spike (volume > 1.5x 20-period average) 
to confirm strength. Long when price > 4h EMA50, RSI crosses above 50, and volume spike present. 
Short when price < 4h EMA50, RSI crosses below 50, and volume spike present. Exit when RSI crosses back to 50.
Target: 80-150 total trades over 4 years (20-37/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 1h RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI crossing signals
    rsi_above_50 = rsi > 50
    rsi_below_50 = rsi < 50
    rsi_cross_up = rsi_above_50 & (np.r_[False, rsi[:-1] <= 50])
    rsi_cross_down = rsi_below_50 & (np.r_[False, rsi[:-1] >= 50])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: trend + momentum + volume
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        
        long_entry = price_above_ema and rsi_cross_up[i] and vol_confirm
        short_entry = price_below_ema and rsi_cross_down[i] and vol_confirm
        
        # Exit when RSI crosses back to 50 (mean reversion)
        exit_long = position == 1 and rsi_cross_down[i]
        exit_short = position == -1 and rsi_cross_up[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_ema_rsi_vol"
timeframe = "1h"
leverage = 1.0