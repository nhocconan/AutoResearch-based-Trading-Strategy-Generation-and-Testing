#!/usr/bin/env python3
"""
6h_1w_1d_camarilla_breakout_volume_v1
Strategy: 6h Camarilla breakout with weekly trend filter and daily volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses weekly Camarilla levels for structural support/resistance, filtered by 1d EMA50 trend direction, with daily volume spike confirmation. Designed to capture high-probability breakouts in trending markets while avoiding false signals in chop. Weekly timeframe provides strong trend filter for 6b entries, reducing false breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
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
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly Camarilla levels (based on previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla H4/L4 levels: Close +/- 1.1*(High-Low)/2
    camarilla_H4_1w = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_L4_1w = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_H4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H4_1w)
    camarilla_L4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L4_1w)
    
    # Daily volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or
            np.isnan(camarilla_H4_1w_aligned[i]) or np.isnan(camarilla_L4_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price_close > ema_50_1d_aligned[i]
        downtrend = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions using weekly Camarilla levels
        breakout_up = price_close > camarilla_H4_1w_aligned[i]
        breakout_down = price_close < camarilla_L4_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend
        
        # Exit when price returns to the opposite Camarilla level
        exit_long = position == 1 and price_close < camarilla_L4_1w_aligned[i]
        exit_short = position == -1 and price_close > camarilla_H4_1w_aligned[i]
        
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