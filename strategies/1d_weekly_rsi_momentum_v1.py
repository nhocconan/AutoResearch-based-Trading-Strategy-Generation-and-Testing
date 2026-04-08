#!/usr/bin/env python3
"""
1d_weekly_rsi_momentum_v1
Hypothesis: Weekly RSI momentum combined with daily price action provides edge in both bull and bear markets.
Long when weekly RSI > 50 and daily close breaks above daily EMA20 with volume confirmation.
Short when weekly RSI < 50 and daily close breaks below daily EMA20 with volume confirmation.
Uses weekly RSI as trend filter to avoid counter-trend trades, reducing whipsaw in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Weekly data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly RSI calculation
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[:13] = np.nan  # Not enough data
    
    # Daily EMA20 for trend
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        rsi_weekly = rsi_1w_aligned[i]
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: weekly RSI < 45 or price below EMA20
            if rsi_weekly < 45 or close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly RSI > 55 or price above EMA20
            if rsi_weekly > 55 or close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: weekly RSI > 50, price above EMA20, volume confirmation
            if rsi_weekly > 50 and close[i] > ema_20[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: weekly RSI < 50, price below EMA20, volume confirmation
            elif rsi_weekly < 50 and close[i] < ema_20[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals