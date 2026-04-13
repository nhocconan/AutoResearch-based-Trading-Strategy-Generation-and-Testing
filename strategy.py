#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d Camarilla pivot breakout with volume confirmation and RSI filter
# Strategy trades breakouts from Camarilla levels (H3/L3) on 4h timeframe
# Uses 1d pivot calculation for structural levels and RSI(14) for momentum filter
# Volume confirmation ensures breakout validity
# Target: 20-50 trades per year (80-200 total) for 4h timeframe
# Works in bull/bear markets via breakout logic + volume filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    h3 = close_1d + range_hl * 1.1 / 4
    l3 = close_1d - range_hl * 1.1 / 4
    h4 = close_1d + range_hl * 1.1 / 2
    l4 = close_1d - range_hl * 1.1 / 2
    
    # Align levels to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # RSI(14) for momentum filter
    def calculate_rsi(close_prices, period=14):
        rsi = np.full_like(close_prices, np.nan, dtype=np.float64)
        if len(close_prices) < period + 1:
            return rsi
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi[period:] = 100 - (100 / (1 + rs[period-1:]))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation (20-period average)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi_val = rsi[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Price breaks above H3 with volume confirmation and RSI > 50
            if (price > h3_aligned[i] and volume_confirm and rsi_val > 50):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below L3 with volume confirmation and RSI < 50
            elif (price < l3_aligned[i] and volume_confirm and rsi_val < 50):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below L3 or RSI < 30
            if (price < l3_aligned[i] or rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above H3 or RSI > 70
            if (price > h3_aligned[i] or rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Breakout_Volume_RSI"
timeframe = "4h"
leverage = 1.0