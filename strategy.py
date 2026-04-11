#!/usr/bin/env python3
"""
4h_1d_camarilla_pivot_volume_trend_v1
Strategy: 4h Camarilla pivot levels from 1d with volume confirmation and trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Combines Camarilla pivot levels (R4/H4 and S4/L4) from daily timeframe as strong support/resistance levels, requiring volume confirmation (>1.5x average volume) and filtered by 1d EMA50 trend alignment. Works in both bull and bear markets by following the higher timeframe trend. Targets 20-50 trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # These are strong intraday support/resistance levels
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (1.5 * range_1d)
    camarilla_l4 = close_1d - (1.5 * range_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Price touching Camarilla levels with some tolerance
        # Using 0.1% tolerance to avoid whipsaws
        tolerance = 0.001
        near_h4 = abs(price_close - camarilla_h4_aligned[i]) < (tolerance * camarilla_h4_aligned[i])
        near_l4 = abs(price_close - camarilla_l4_aligned[i]) < (tolerance * camarilla_l4_aligned[i])
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: price near L4 support with volume in uptrend (bounce)
        long_signal = near_l4 and vol_confirmed and uptrend_1d
        
        # Short: price near H4 resistance with volume in downtrend (rejection)
        short_signal = near_h4 and vol_confirmed and downtrend_1d
        
        # Exit when price moves back toward the daily close (pivot point)
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        exit_long = position == 1 and price_close > daily_close_aligned[i]
        exit_short = position == -1 and price_close < daily_close_aligned[i]
        
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