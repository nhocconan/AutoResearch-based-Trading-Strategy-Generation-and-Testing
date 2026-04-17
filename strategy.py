#!/usr/bin/env python3
"""
12h_WAVES_1W_Volume_Signal_v1
Trades weekly trend waves using 1w EMA13 as trend filter and 12h price action.
Long when price > 1w EMA13 + 12h close > 12h open + volume spike.
Short when price < 1w EMA13 + 12h close < 12h open + volume spike.
Exit on opposite signal or trend change.
Target: 60-120 total trades over 4 years (15-30/year).
Works in bull via trend-following, bear via shorting weakness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Volume spike detection (12h) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 * 12h = 12 days
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1w EMA13 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_13_1w = pd.Series(df_1w['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike threshold
        vol_spike = vol_ratio[i] > 2.0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: above 1w EMA13, bullish 12h candle, volume spike
            if (close[i] > ema_13_1w_aligned[i] and 
                close[i] > open_price[i] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                continue
            # Short: below 1w EMA13, bearish 12h candle, volume spike
            elif (close[i] < ema_13_1w_aligned[i] and 
                  close[i] < open_price[i] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price below 1w EMA13 OR bearish 12h candle with volume
            if (close[i] < ema_13_1w_aligned[i] or 
                (close[i] < open_price[i] and vol_spike)):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1w EMA13 OR bullish 12h candle with volume
            if (close[i] > ema_13_1w_aligned[i] or 
                (close[i] > open_price[i] and vol_spike)):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WAVES_1W_Volume_Signal_v1"
timeframe = "12h"
leverage = 1.0