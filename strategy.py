#!/usr/bin/env python3
"""
6h_Liquidity_Grab_Reversal
Hypothesis: In ranging markets, price often triggers liquidity sweeps (false breakouts) at prior session highs/lows before reversing. This strategy fades these moves using 1-day session extremes as liquidity zones, confirmed by RSI divergence and volume exhaustion. Works in bull/bear by capturing mean reversion at key levels.
"""

name = "6h_Liquidity_Grab_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for session high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day RSI for divergence
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = np.zeros_like(gain_1d)
    avg_loss_1d = np.zeros_like(loss_1d)
    if len(gain_1d) >= 14:
        avg_gain_1d[13] = np.mean(gain_1d[1:14])
        avg_loss_1d[13] = np.mean(loss_1d[1:14])
        for i in range(14, len(gain_1d)):
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    
    rs_1d = np.divide(avg_gain_1d, avg_loss_1d, out=np.zeros_like(avg_gain_1d), where=avg_loss_1d!=0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    
    # Align 1d data to 6h
    session_high_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    session_low_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate volume exhaustion (decreasing volume on move)
    vol_ma_10 = np.zeros_like(volume)
    for i in range(9, len(volume)):
        vol_ma_10[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(session_high_aligned[i]) or 
            np.isnan(session_low_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Liquidity grab conditions
        near_session_high = high[i] >= session_high_aligned[i] * 0.999  # Within 0.1% of session high
        near_session_low = low[i] <= session_low_aligned[i] * 1.001   # Within 0.1% of session low
        
        # RSI divergence: price makes new high/low but RSI doesn't
        rsi_divergence_high = near_session_high and rsi[i] < rsi[i-1] and close[i] > close[i-1]
        rsi_divergence_low = near_session_low and rsi[i] > rsi[i-1] and close[i] < close[i-1]
        
        # Volume exhaustion: current volume < average volume
        volume_exhaustion = volume[i] < vol_ma_10[i]
        
        if position == 0:
            # LONG: liquidity grab at session low + RSI bearish divergence + volume exhaustion
            if near_session_low and rsi_divergence_low and volume_exhaustion:
                signals[i] = 0.25
                position = 1
            # SHORT: liquidity grab at session high + RSI bullish divergence + volume exhaustion
            elif near_session_high and rsi_divergence_high and volume_exhaustion:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to session low or RSI recovers
            if close[i] <= session_low_aligned[i] * 1.002 or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to session high or RSI recovers
            if close[i] >= session_high_aligned[i] * 0.998 or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals