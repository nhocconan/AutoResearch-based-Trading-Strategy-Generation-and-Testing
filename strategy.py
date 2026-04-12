#!/usr/bin/env python3
"""
6h_1w_RSI_Extremes_v1
Hypothesis: Use weekly RSI extremes with 6h price action to capture mean reversion in overbought/oversold conditions.
Long when weekly RSI < 30 and 6h price closes above 6h VWAP, short when weekly RSI > 70 and 6h price closes below 6h VWAP.
Exit when weekly RSI returns to neutral zone (40-60). Weekly RSI avoids whipsaw from lower timeframe noise.
Works in bull via buying oversold dips, in bear via selling overbought rallies.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_RSI_Extremes_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly RSI calculation (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:  # Need at least 14 periods for RSI
        return np.zeros(n)
    
    # Calculate weekly RSI
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 gains
    avg_loss[13] = np.mean(loss[1:14])  # First average of first 14 losses
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    rsi_1w = np.where(avg_gain == 0, 0, rsi_1w)  # Handle case where avg_gain is 0
    
    # Align weekly RSI to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 6h VWAP calculation (typical price * volume cumsum / volume cumsum)
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: weekly RSI extremes + price vs VWAP
        long_entry = rsi_1w_aligned[i] < 30 and close[i] > vwap[i]
        short_entry = rsi_1w_aligned[i] > 70 and close[i] < vwap[i]
        
        # Exit conditions: weekly RSI returns to neutral zone (40-60)
        long_exit = rsi_1w_aligned[i] >= 40
        short_exit = rsi_1w_aligned[i] <= 60
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals