#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot reversal with 4h trend filter and volume confirmation.
# Camarilla pivots identify high-probability reversal levels in ranging markets.
# 4h trend filter ensures we only take reversals in the direction of higher timeframe trend.
# Volume confirmation reduces false signals.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Works in both bull and bear markets by trading mean reversions within the trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # We'll use daily data to calculate pivots, then apply to 1h
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily pivot points for Camarilla
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.zeros(len(close_1d))  # Resistance 4
    camarilla_l4 = np.zeros(len(close_1d))  # Support 4
    camarilla_h3 = np.zeros(len(close_1d))  # Resistance 3
    camarilla_l3 = np.zeros(len(close_1d))  # Support 3
    
    for i in range(len(close_1d)):
        if i == 0:
            # For first day, use current day's data (not ideal but avoids NaN)
            rng = high_1d[i] - low_1d[i]
        else:
            rng = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i] + 1.5 * rng
        camarilla_l4[i] = close_1d[i] - 1.5 * rng
        camarilla_h3[i] = close_1d[i] + 1.25 * rng
        camarilla_l3[i] = close_1d[i] - 1.25 * rng
    
    # Align Camarilla levels to 1h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation (20-period average)
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_4h = ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price rejects at L3/L4 support + volume + price above 4h EMA (uptrend)
            if (price <= l3_aligned[i] or price <= l4_aligned[i]) and volume_confirm and price > ema_4h:
                position = 1
                signals[i] = position_size
            # Short: price rejects at H3/H4 resistance + volume + price below 4h EMA (downtrend)
            elif (price >= h3_aligned[i] or price >= h4_aligned[i]) and volume_confirm and price < ema_4h:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches opposite H3 level or volume drops significantly
            if price >= h3_aligned[i] or vol < 0.5 * avg_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches opposite L3 level or volume drops significantly
            if price <= l3_aligned[i] or vol < 0.5 * avg_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_Camarilla_Pivot_Reversal_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0