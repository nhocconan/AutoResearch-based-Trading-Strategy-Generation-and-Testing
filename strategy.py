#!/usr/bin/env python3
# [24916] 12h_1d_camarilla_pivot_v1
# Hypothesis: 12-hour Camarilla pivot levels from 1-day timeframe with volume confirmation.
# Long when price crosses above H4 resistance level with volume > 1.5x average and RSI(14) < 70.
# Short when price crosses below L4 support level with volume > 1.5x average and RSI(14) > 30.
# Exit when price returns to Pivot Point level or volume drops below 1.2x average.
# Uses Camarilla pivot levels derived from prior day's range for intraday support/resistance.
# Works in both bull and bear markets by fading extremes at key levels with volume confirmation.
# Target: 15-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily high, low, close for pivot
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = P + 1.1 * R / 2
    # L4 = P - 1.1 * R / 2
    # H3 = P + 1.1 * R / 4
    # L3 = P - 1.1 * R / 4
    # H2 = P + 1.1 * R / 6
    # L2 = P - 1.1 * R / 6
    # H1 = P + 1.1 * R / 12
    # L1 = P - 1.1 * R / 12
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    rang = daily_high - daily_low
    
    H4 = pivot + 1.1 * rang / 2.0
    L4 = pivot - 1.1 * rang / 2.0
    H3 = pivot + 1.1 * rang / 4.0
    L3 = pivot - 1.1 * rang / 4.0
    
    # Align Camarilla levels to 12-hour timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate RSI(14) on 12-hour timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:15])  # First average of gains
        avg_loss[13] = np.mean(loss[1:15])  # First average of losses
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to pivot or volume drops below 1.2x average
            if price <= pivot_aligned[i] or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to pivot or volume drops below 1.2x average
            if price >= pivot_aligned[i] or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price crosses above H4 with volume expansion and not overbought
            if price > H4_aligned[i] and vol_ratio > 1.5 and rsi[i] < 70:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below L4 with volume expansion and not oversold
            elif price < L4_aligned[i] and vol_ratio > 1.5 and rsi[i] > 30:
                position = -1
                signals[i] = -0.25
    
    return signals