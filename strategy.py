#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean-reversion with 4h trend filter and volume confirmation.
Long when RSI < 30 (oversold), 4h EMA50 rising, and volume spike.
Short when RSI > 70 (overbought), 4h EMA50 falling, and volume spike.
Exit when RSI crosses 50 (mean reversion complete) or trend reverses.
Uses 4h for trend direction to avoid counter-trend trades, 1h for precise entry timing.
Session filter (08-20 UTC) reduces noise. Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 4h trend.
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
    
    # RSI(14) - standard calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 4h close for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(rsi[i]) or not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold), 4h EMA50 rising, volume spike
            if (rsi[i] < 30 and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and vol_spike):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought), 4h EMA50 falling, volume spike
            elif (rsi[i] > 70 and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and vol_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI crosses 50 (mean reversion complete) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI >= 50 or 4h EMA50 turns down
                if rsi[i] >= 50 or ema50_4h_aligned[i] < ema50_4h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI <= 50 or 4h EMA50 turns up
                if rsi[i] <= 50 or ema50_4h_aligned[i] > ema50_4h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0