#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Momentum
Strategy: 4h Camarilla R1/S1 breakout with 12h momentum filter and volume confirmation.
Long: Price breaks above R1 + 12h RSI > 50 + volume > 1.3x 20-period average
Short: Price breaks below S1 + 12h RSI < 50 + volume > 1.3x 20-period average
Exit: Opposite breakout or RSI reversal
Position size: 0.25
Uses Camarilla pivot levels for intraday support/resistance, 12h RSI for momentum filter, volume for confirmation.
Designed to work in both bull and bear markets by requiring momentum alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for momentum filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI(14) for momentum filter
    delta = pd.Series(close_12h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_12h = (100 - (100 / (1 + rs))).values
    rsi_14_12h[:14] = np.nan  # Set first 14 values to NaN
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = Close + (High - Low) * 1.1/12, S1 = Close - (High - Low) * 1.1/12
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    s1 = close_1d - rng * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h volume average (20-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(rsi_14_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume aligned to 4h
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.3 * volume_ma20_4h_aligned[i])
        
        # Momentum filter: 12h RSI > 50 for long, < 50 for short
        rsi_long = rsi_14_12h_aligned[i] > 50
        rsi_short = rsi_14_12h_aligned[i] < 50
        
        # Camarilla breakout signals
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: Camarilla R1 breakout + 12h RSI > 50 + volume filter
            if breakout_up and rsi_long and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S1 breakout + 12h RSI < 50 + volume filter
            elif breakout_down and rsi_short and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Camarilla S1 break or RSI < 40
            if breakout_down or rsi_14_12h_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Camarilla R1 break or RSI > 60
            if breakout_up or rsi_14_12h_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Momentum"
timeframe = "4h"
leverage = 1.0