#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d trend alignment and volume confirmation.
# Long: 1h close above 4h EMA(21) AND 1h close above 1d VWAP AND volume > 1.5x 20-period average volume.
# Short: 1h close below 4h EMA(21) AND 1h close below 1d VWAP AND volume > 1.5x 20-period average volume.
# Exit: When price crosses back below/above the 4h EMA(21).
# Uses 4h EMA for trend direction, 1d VWAP for institutional reference, volume for conviction.
# Time filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA(21)
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 21:
        ema_4h[20] = np.mean(close_4h[:21])  # Simple average for first value
        for i in range(21, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 19) / 21  # EMA formula
    
    # Calculate 1d VWAP (volume-weighted average price)
    vwap_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i == 0:
            vwap_1d[i] = (df_1d['close'].iloc[i] * df_1d['volume'].iloc[i]) / df_1d['volume'].iloc[i]
        else:
            typical_price = (df_1d['high'].iloc[i] + df_1d['low'].iloc[i] + df_1d['close'].iloc[i]) / 3
            vwap_1d[i] = (vwap_1d[i-1] * df_1d['volume'].iloc[i-1] + typical_price * df_1d['volume'].iloc[i]) / (df_1d['volume'].iloc[i-1] + df_1d['volume'].iloc[i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 4h EMA and 1d VWAP to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    # Pre-calculate hour filter for performance
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema = ema_4h_aligned[i]
        vwap = vwap_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price above both EMA and VWAP + volume confirmation
            if (price > ema and price > vwap and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price below both EMA and VWAP + volume confirmation
            elif (price < ema and price < vwap and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below EMA (trend change)
            if price < ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above EMA (trend change)
            if price > ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_EMA_VWAP_Volume_Filter"
timeframe = "1h"
leverage = 1.0