#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
- Long: Close breaks above Camarilla R3 + price > 1w EMA34 (uptrend) + volume > 1.5x 20-day average
- Short: Close breaks below Camarilla S3 + price < 1w EMA34 (downtrend) + volume > 1.5x 20-day average
- Exit: Close retouches Camarilla H3/L3 level OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 15-25 trades/year (60-100 over 4 years) to avoid fee drag
- Daily timeframe reduces noise; weekly EMA34 provides strong trend filter; Camarilla levels from daily OHLC work in both bull and bear markets
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from 1d OHLC (previous day's values)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 1d timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)  # 1d data needs 50, 1w EMA needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1w EMA34
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: Close breaks above H3 + uptrend + volume spike
        # Short: Close breaks below L3 + downtrend + volume spike
        long_signal = (close[i] > h3_aligned[i] and 
                      uptrend and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < l3_aligned[i] and 
                       downtrend and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Close retouches H3/L3 level OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Close retouches H3 level or trend turns down
                if (close[i] <= h3_aligned[i] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Close retouches L3 level or trend turns up
                if (close[i] >= l3_aligned[i] or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0