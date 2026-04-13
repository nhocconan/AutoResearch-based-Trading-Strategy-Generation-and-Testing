#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w exponential moving average (EMA) trend filter and volume confirmation.
# Long: Price crosses above 1w EMA(34) + volume > 1.5x average volume (30-period).
# Short: Price crosses below 1w EMA(34) + volume > 1.5x average volume.
# Uses 1w EMA for long-term trend direction, 1d for execution with volume confirmation.
# Volume filter prevents whipsaws in choppy markets.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on weekly close
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])  # SMA for first value
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_34_1w[i-1]
    
    # Average volume (30-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(30, n):
        avg_volume[i] = np.mean(volume[i-30:i])
    
    # Align 1w EMA to 1d
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_34 = ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price crosses above EMA + volume confirmation
            if (price > ema_34 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price crosses below EMA + volume confirmation
            elif (price < ema_34 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below EMA
            if price < ema_34:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above EMA
            if price > ema_34:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA34_Volume_Filter"
timeframe = "1d"
leverage = 1.0