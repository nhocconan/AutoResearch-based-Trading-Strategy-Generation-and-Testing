#!/usr/bin/env python3
"""
4h_Long_Only_Pullback_v1
Hypothesis: Long-only strategy that buys pullbacks in uptrends using 1d EMA200 as trend filter.
Enters on RSI(14) dip below 35 with volume confirmation, exits on RSI > 65 or trend reversal.
Designed for low frequency (15-25 trades/year) to work in bull markets and avoid whipsaws in bear markets by only taking longs in uptrends.
"""

name = "4h_Long_Only_Pullback_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA200 for trend filter ---
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # --- RSI(14) ---
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Only take longs in uptrend (price above 1d EMA200)
        uptrend = close[i] > ema_200_1d_aligned[i]
        
        # Entry: RSI pullback with volume confirmation in uptrend
        rsi_pullback = rsi_values[i] < 35
        entry_signal = uptrend and rsi_pullback and vol_spike[i]
        
        # Exit: RSI > 65 or trend reversal
        exit_signal = (rsi_values[i] > 65) or (not uptrend)
        
        if position == 0:
            if entry_signal:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        else:  # position == 1
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals