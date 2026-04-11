#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v1
# Strategy: 12-hour Camarilla pivot breakout with 1-day trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breaks key Camarilla levels (H3/L3) on the 12h chart with 1-day trend alignment
# and volume confirmation. Works in both bull and bear by trading breakouts in the direction of
# the higher timeframe trend. Uses volume to filter false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
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
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # H3 = close + (high - low) * 1.1/2
    # L3 = close - (high - low) * 1.1/2
    camarilla_height = (high_1d - low_1d) * 1.1 / 2.0
    camarilla_h3 = close_1d + camarilla_height
    camarilla_l3 = close_1d - camarilla_height
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout signals: price breaks Camarilla H3/L3 with volume
        long_signal = (price_close > camarilla_h3_aligned[i]) and vol_spike[i] and uptrend_1d
        short_signal = (price_close < camarilla_l3_aligned[i]) and vol_spike[i] and downtrend_1d
        
        # Exit when price returns to midpoint of the range or opposite breakout
        camarilla_mid = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2.0
        exit_long = position == 1 and (price_close < camarilla_mid)
        exit_short = position == -1 and (price_close > camarilla_mid)
        
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