#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 1d ATR and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1d EMA (34-period)
    ema_34 = np.zeros_like(close_1d)
    ema_34[0] = close_1d[0]
    alpha = 2 / (34 + 1)
    for i in range(1, len(close_1d)):
        ema_34[i] = alpha * close_1d[i] + (1 - alpha) * ema_34[i-1]
    
    # Align 1d indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 12h Donchian channel (20-period)
    dc_upper = np.zeros(n)
    dc_lower = np.zeros(n)
    for i in range(20, n):
        dc_upper[i] = np.max(high[i-20:i])
        dc_lower[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.8x 20-period average
    volume_ma = np.zeros(n)
    for i in range(20, n):
        volume_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(atr_14_aligned[i]) or np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * 20-period average ATR
        if i >= 40:
            atr_ma = np.mean(atr_14_aligned[i-20:i])
            vol_filter = atr_14_aligned[i] > 0.5 * atr_ma
        else:
            vol_filter = True
        
        # Trend filter: price > EMA34 for long, price < EMA34 for short
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        # Entry conditions
        long_entry = (close[i] > dc_upper[i]) and price_above_ema and vol_filter and volume_filter[i]
        short_entry = (close[i] < dc_lower[i]) and price_below_ema and vol_filter and volume_filter[i]
        
        # Exit conditions: ATR-based stop loss
        long_exit = position == 1 and close[i] < (dc_upper[i] - 1.5 * atr_14_aligned[i])
        short_exit = position == -1 and close[i] > (dc_lower[i] + 1.5 * atr_14_aligned[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_DonchianBreakout_1dATREMA_VolumeFilter"
timeframe = "12h"
leverage = 1.0