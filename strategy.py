#!/usr/bin/env python3
"""
4h_1d_camarilla_pivot_volume_v1
Strategy: 4h Camarilla pivot levels with volume confirmation and 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses daily Camarilla pivot levels (H4, L4, H6, L6) as key support/resistance levels.
Enters long when price breaks above H4 with volume confirmation in a 1d uptrend, and short when price breaks below L4 with volume confirmation in a 1d downtrend.
Exits when price returns to the pivot point (P). Uses volume spike (>1.5x average) to filter false breakouts.
Designed to work in both bull and bear markets by following the 1d trend direction.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
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
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA20 for dynamic exit (optional, can use pivot point)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # Calculate daily Camarilla pivot levels from previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H6 = Close + 2.0 * (High - Low)
    # L6 = Close - 2.0 * (High - Low)
    # P = (High + Low + Close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_p = (high_1d + low_1d + close_1d) / 3.0
    
    # Align to 4h timeframe (wait for daily bar to close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) if i >= 20 else True) or \
           np.isnan(camarilla_h4_aligned[i]) or \
           np.isnan(camarilla_l4_aligned[i]) or \
           np.isnan(camarilla_p_aligned[i]) or \
           np.isnan(vol_avg[i]) or \
           np.isnan(ema_50_1d_aligned[i]):
            # Hold current position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_h4_aligned[i-1]  # Use previous bar's level
        breakout_down = price_close < camarilla_l4_aligned[i-1]  # Use previous bar's level
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1d
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1d
        
        # Exit when price returns to pivot point (P)
        exit_long = position == 1 and price_close < camarilla_p_aligned[i]
        exit_short = position == -1 and price_close > camarilla_p_aligned[i]
        
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