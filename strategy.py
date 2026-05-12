#!/usr/bin/env python3
"""
6H_WEEKLY_PIVOT_REVERSAL_WITH_VOLUME_CONFIRMATION
Hypothesis: Price reverses at weekly pivot levels (S3/R3) with volume exhaustion signals.
In bull markets: buy at S3 with volume drying up on down moves.
In bear markets: sell at R3 with volume drying up on up moves.
Uses volume divergence (current volume < 50% of 20-period average) as exhaustion filter.
Targets 15-35 trades/year with controlled risk via weekly pivot structure.
"""
name = "6H_WEEKLY_PIVOT_REVERSAL_WITH_VOLUME_CONFIRMATION"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    # Pivot = (H + L + C) / 3
    # S1 = 2*P - H, S2 = P - (H - L), S3 = P - 2*(H - L)
    # R1 = 2*P - L, R2 = P + (H - L), R3 = P + 2*(H - L)
    wk_high = df_w['high'].values
    wk_low = df_w['low'].values
    wk_close = df_w['close'].values
    
    pivot = (wk_high + wk_low + wk_close) / 3
    s3 = pivot - 2 * (wk_high - wk_low)
    r3 = pivot + 2 * (wk_high - wk_low)
    
    # Align weekly pivot levels to 6h timeframe (1 week = 28 * 6h bars)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    
    # Volume exhaustion: current volume < 50% of 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20  # Current volume / 20-period average
    
    # RSI for overbought/oversold confirmation (14-period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price at or below S3 with RSI oversold and volume exhaustion
            if (low[i] <= s3_aligned[i] and 
                rsi_values[i] < 30 and 
                vol_ratio[i] < 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or above R3 with RSI overbought and volume exhaustion
            elif (high[i] >= r3_aligned[i] and 
                  rsi_values[i] > 70 and 
                  vol_ratio[i] < 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above pivot or RSI overbought
            if close[i] > pivot[i] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below pivot or RSI oversold
            if close[i] < pivot[i] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals