#!/usr/bin/env python3
"""
4h_Triple_RSI_Trend_With_Volume
Hypothesis: RSI(14) combined with EMA trend and volume confirmation creates robust entries.
Long when RSI crosses above 50 with EMA9 > EMA21 and volume spike.
Short when RSI crosses below 50 with EMA9 < EMA21 and volume spike.
Exit when RSI reverts to opposite extreme (30/70) or trend fails.
Uses 12h EMA200 for higher timeframe trend filter.
Designed for 4h timeframe to balance trade frequency and signal quality.
Targets 20-40 trades/year (80-160 over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align length
    
    # EMA9 and EMA21 for trend
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike detection (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 12h EMA200 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema200_12h = pd.Series(df_12h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data invalid
        if (np.isnan(rsi[i]) or np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema200_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema9_val = ema9[i]
        ema21_val = ema21[i]
        vol_spike = volume_spike[i]
        ema200_12h_val = ema200_12h_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long
            # Exit: RSI < 30 (oversold) OR trend fails (EMA9 < EMA21) OR price below 12h EMA200
            if rsi_val < 30 or ema9_val < ema21_val or price < ema200_12h_val:
                signals[i] = 0.0
                position = 0
                continue
                
        elif position == -1:  # Short
            # Exit: RSI > 70 (overbought) OR trend fails (EMA9 > EMA21) OR price above 12h EMA200
            if rsi_val > 70 or ema9_val > ema21_val or price > ema200_12h_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # Long: RSI > 50, EMA9 > EMA21, volume spike, price above 12h EMA200 (uptrend)
            if rsi_val > 50 and ema9_val > ema21_val and vol_spike and price > ema200_12h_val:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: RSI < 50, EMA9 < EMA21, volume spike, price below 12h EMA200 (downtrend)
            elif rsi_val < 50 and ema9_val < ema21_val and vol_spike and price < ema200_12h_val:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Triple_RSI_Trend_With_Volume"
timeframe = "4h"
leverage = 1.0