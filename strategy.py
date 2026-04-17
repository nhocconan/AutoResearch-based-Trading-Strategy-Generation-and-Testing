#!/usr/bin/env python3
"""
Hypothesis: On 1h timeframe, price often respects 4-hour and daily key levels (S2/R2 from Camarilla).
By combining 4h Camarilla S2/R2 with 1d EMA50 trend filter and volume spikes (>2x 20-bar avg),
we capture high-probability breakouts. Use 4h/1d for signal direction, 1h only for entry timing.
Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year (~60-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels from previous 4h bar
    phigh = df_4h['high'].values
    plow = df_4h['low'].values
    pclose = df_4h['close'].values
    
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Camarilla S2/R2 levels
    R2 = pivot + (range_ * 1.1 / 6)
    S2 = pivot - (range_ * 1.1 / 6)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all levels to 1h timeframe
    R2_1h = align_htf_to_ltf(prices, df_4h, R2)
    S2_1h = align_htf_to_ltf(prices, df_4h, S2)
    pivot_1h = align_htf_to_ltf(prices, df_4h, pivot)
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 1h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(R2_1h[i]) or np.isnan(S2_1h[i]) or np.isnan(pivot_1h[i]) or
            np.isnan(ema_50_1h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0 and in_session:
            # Long: break above R2 with volume spike and above 1d EMA50
            if price > R2_1h[i] and vol > 2.0 * vol_ma and price > ema_50_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below S2 with volume spike and below 1d EMA50
            elif price < S2_1h[i] and vol > 2.0 * vol_ma and price < ema_50_1h[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or closes below session
            if price < pivot_1h[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns to pivot or closes below session
            if price > pivot_1h[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_S2R2_Volume_EMA50_Session"
timeframe = "1h"
leverage = 1.0