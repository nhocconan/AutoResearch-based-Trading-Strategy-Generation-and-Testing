#!/usr/bin/env python3
"""
1h_Liquidity_Zone_Reversal_4hTrend
Hypothesis: Price often reverses at prior session highs/lows (liquidity zones) during 1h sessions.
Trades in direction of 4h trend (EMA50) with entries at 1h swing points confirmed by volume.
Designed for ranging markets (2025) while capturing trend continuations. Target 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 4h trend filter: EMA50 ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h swing points (pivot highs/lows) ===
    high = prices['high'].values
    low = prices['low'].values
    # Pivot high: higher high on both sides
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    for i in range(2, n-2):
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            ph[i] = True
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            pl[i] = True
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        price_close = prices['close'].iloc[i]
        trend_4h = ema_50_4h_aligned[i]
        
        if position == 0 and in_session:
            # Long at pivot low with volume spike and above 4h EMA50
            if pl[i] and vol_ratio[i] > 1.5 and price_close > trend_4h:
                signals[i] = 0.20
                position = 1
            # Short at pivot high with volume spike and below 4h EMA50
            elif ph[i] and vol_ratio[i] > 1.5 and price_close < trend_4h:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: opposite pivot or against trend
            exit_signal = False
            if position == 1:
                if ph[i] or price_close < trend_4h:
                    exit_signal = True
            else:  # position == -1
                if pl[i] or price_close > trend_4h:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Liquidity_Zone_Reversal_4hTrend"
timeframe = "1h"
leverage = 1.0