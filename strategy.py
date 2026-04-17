#!/usr/bin/env python3
"""
Hypothesis: On the 6-hour timeframe, price reverses from extreme weekly RSI levels when confirmed by daily volume imbalance.
We use weekly RSI(14) for overbought/oversold conditions and daily volume delta (buy vs sell volume) for confirmation.
Long when weekly RSI < 30 (oversold) and daily volume delta > 0 (buying pressure).
Short when weekly RSI > 70 (overbought) and daily volume delta < 0 (selling pressure).
Exit when RSI returns to neutral zone (40-60) or on opposite signal.
Designed for 6h to work in ranging markets with mean reversion from extremes, suitable for both accumulation and distribution phases.
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
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI(14)
    delta = np.diff(df_1w['close'].values, prepend=df_1w['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get daily data for volume delta calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily volume delta (buy volume - sell volume)
    buy_volume = df_1d['taker_buy_volume'].values
    sell_volume = df_1d['volume'].values - buy_volume
    volume_delta = buy_volume - sell_volume
    
    # Align weekly RSI to 6h timeframe (waits for weekly bar to close)
    rsi_6h = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # Align daily volume delta to 6h timeframe (waits for daily bar to close)
    volume_delta_6h = align_htf_to_ltf(prices, df_1d, volume_delta)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for RSI calculation
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_6h[i]) or np.isnan(volume_delta_6h[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_6h[i]
        vol_delta = volume_delta_6h[i]
        
        if position == 0:
            # Long: weekly RSI oversold (<30) with buying pressure (positive volume delta)
            if rsi_val < 30 and vol_delta > 0:
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI overbought (>70) with selling pressure (negative volume delta)
            elif rsi_val > 70 and vol_delta < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral zone (>=40) or opposite signal
            if rsi_val >= 40 or (rsi_val > 70 and vol_delta < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral zone (<=60) or opposite signal
            if rsi_val <= 60 or (rsi_val < 30 and vol_delta > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyRSI_VolumeDelta_MeanReversion"
timeframe = "6h"
leverage = 1.0