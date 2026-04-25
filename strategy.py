#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1wTrend_VolumeSpike
Hypothesis: 6-hour Camarilla R4/S4 breakout with 1-week EMA50 trend filter and volume spike confirmation.
Long when price breaks above R4 in 1-week uptrend (close > 1w EMA50) with volume > 2.5x 20-period average.
Short when price breaks below S4 in 1-week downtrend (close < 1w EMA50) with volume > 2.5x 20-period average.
Exit via opposite Camarilla level (S4 for longs, R4 for shorts).
Designed for ~12-37 trades/year via tight R4/S4 breakout conditions and volume confirmation.
Uses 1-week trend filter to work in both bull and bear markets, avoiding false breakouts via volume confirmation.
"""

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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume regime: volume > 2.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.5 * vol_ma_20)
    
    # Get weekly OHLC for Camarilla levels (use same 1w data)
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Calculate Camarilla levels (R4/S4)
    # R4 = c + (h-l)*1.1/2
    # S4 = c - (h-l)*1.1/2
    camarilla_r4_1w = c_1w + ((h_1w - l_1w) * 1.1 / 2)
    camarilla_s4_1w = c_1w - ((h_1w - l_1w) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        r4_level = camarilla_r4_aligned[i]
        s4_level = camarilla_s4_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1w EMA50 filter)
            if close[i] > ema_trend:  # 1w uptrend regime
                # Long: break above R4 with volume confirmation
                long_signal = (close[i] > r4_level) and vol_regime[i]
            else:  # 1w downtrend regime
                # Short: break below S4 with volume confirmation
                short_signal = (close[i] < s4_level) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit condition: Price breaks below S4 (opposite Camarilla level)
            if close[i] < s4_level:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit condition: Price breaks above R4 (opposite Camarilla level)
            if close[i] > r4_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0