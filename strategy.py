#!/usr/bin/env python3
"""
4h_12h_camarilla_volume_trend_v1
Strategy: 4h Camarilla pivot breakout with 12h volume confirmation and trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Long when price breaks above Camarilla H3 with 12h volume above average and 12h close > open (uptrend); short when price breaks below L3 with 12h volume above average and 12h close < open (downtrend). Uses 12h trend filter to avoid counter-trend trades. Designed for low-frequency, high-conviction trades in both bull and bear markets. Target: 20-50 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using daily high/low/close from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily data to 4h timeframe
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_4h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    # H4 = Close + 1.5*(High-Low)
    # H3 = Close + 1.1*(High-Low)
    # L3 = Close - 1.1*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    rang = prev_high_4h - prev_low_4h
    h3 = prev_close_4h + 1.1 * rang
    l3 = prev_close_4h - 1.1 * rang
    
    # 12h trend filter: bullish if close > open, bearish if close < open
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    bullish_12h = close_12h > open_12h
    bearish_12h = close_12h < open_12h
    bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_12h)
    bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_12h)
    
    # 12h volume confirmation: volume above 20-period average
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_above_avg = vol_12h > vol_ma_20
    vol_above_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_above_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(bullish_12h_aligned[i]) or np.isnan(bearish_12h_aligned[i]) or
            np.isnan(vol_above_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Breakout conditions
        breakout_long = price_high > h3[i]  # Price breaks above H3
        breakout_short = price_low < l3[i]  # Price breaks below L3
        
        # Entry conditions with confirmation
        long_signal = breakout_long and bullish_12h_aligned[i] and vol_above_avg_aligned[i]
        short_signal = breakout_short and bearish_12h_aligned[i] and vol_above_avg_aligned[i]
        
        # Exit when price returns to median (Camarilla C3-C4 level)
        # Simplified: exit when price crosses back below H3 for longs or above L3 for shorts
        exit_long = position == 1 and price_close < h3[i]
        exit_short = position == -1 and price_close > l3[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals