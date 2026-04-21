#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: Daily KAMA trend direction with RSI momentum and choppiness regime filter.
Long when KAMA rising, RSI>50, and choppy market (CHOP>61.8); short when KAMA falling, RSI<50, and choppy market.
Uses weekly trend alignment via EMA34 on 1w timeframe to avoid counter-trend trades.
Designed for 1d timeframe with 1w HTF trend filter to work in both bull and bear markets by requiring alignment with higher timeframe trend and volatility regime.
Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w EMA34 for trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === KAMA (10-period ER) for trend direction ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=1))
    volatility = np.sum(np.abs(np.diff(close, n=1)).reshape(-1, 1), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.diff(kama, prepend=kama[0])  # positive = rising, negative = falling
    
    # === RSI (14-period) for momentum ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14-period) for regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    atr_14 = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(np.sum(atr_14) / (max_high - min_low)) / np.log10(14), 
                    50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama_dir[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        kama_direction = kama_dir[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Choppiness filter: only trade in choppy markets (CHOP > 61.8 = ranging)
        chop_filter = chop_val > 61.8
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, above weekly EMA34, choppy market
            long_condition = (kama_direction > 0) and (rsi_val > 50) and (price > ema_34_1w_val) and chop_filter
            # Short: KAMA falling, RSI < 50, below weekly EMA34, choppy market
            short_condition = (kama_direction < 0) and (rsi_val < 50) and (price < ema_34_1w_val) and chop_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 2 days to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit conditions: KAMA direction change or RSI extreme reversal
            if position == 1:
                if kama_direction <= 0 or rsi_val < 30:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if kama_direction >= 0 or rsi_val > 70:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0