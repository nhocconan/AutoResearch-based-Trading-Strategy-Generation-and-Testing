# 6h_TRIX_Trend_Reversal_with_Volume
# Hypothesis: TRIX (triple smoothed EMA) detects momentum exhaustion and reversals. 
# Long when TRIX crosses above zero with volume confirmation and price above 1w EMA200 (bullish regime).
# Short when TRIX crosses below zero with volume confirmation and price below 1w EMA200 (bearish regime).
# Works in bull/bear by filtering trades in direction of weekly trend, reducing whipsaws.
# Target: 15-25 trades/year (60-100 total) to minimize fee drag.

name = "6h_TRIX_Trend_Reversal_with_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for regime filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA200 for trend regime
    ema200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema200_1w[i-1]
    
    # TRIX calculation: triple EMA of percent change
    # ROC = (close[t] - close[t-1]) / close[t-1] * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
    
    # Triple EMA of ROC with period 15
    ema1 = np.full_like(roc, np.nan)
    ema2 = np.full_like(roc, np.nan)
    ema3 = np.full_like(roc, np.nan)
    
    if len(roc) >= 15:
        # First EMA
        ema1[14] = np.mean(roc[:15])
        alpha1 = 2 / (15 + 1)
        for i in range(15, len(roc)):
            ema1[i] = alpha1 * roc[i] + (1 - alpha1) * ema1[i-1]
        
        # Second EMA of first EMA
        ema2[29] = np.mean(ema1[15:30])  # Need 15 values of ema1
        alpha2 = 2 / (15 + 1)
        for i in range(30, len(ema1)):
            ema2[i] = alpha2 * ema1[i] + (1 - alpha2) * ema2[i-1]
        
        # Third EMA of second EMA
        ema3[44] = np.mean(ema2[30:45])  # Need 15 values of ema2
        alpha3 = 2 / (15 + 1)
        for i in range(45, len(ema2)):
            ema3[i] = alpha3 * ema2[i] + (1 - alpha3) * ema3[i-1]
    
    # Align 1w EMA200 to 6h
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 45  # Wait for TRIX calculation
    
    for i in range(start_idx, n):
        if np.isnan(ema3[i]) or np.isnan(ema200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume over last 20 periods
        vol_ma20 = np.full(n, np.nan)
        if i >= 20:
            vol_ma20[i] = np.mean(volume[i-20:i])
        volume_confirm = volume[i] > 1.5 * vol_ma20[i] if not np.isnan(vol_ma20[i]) else False
        
        # TRIX signal: cross above/below zero
        trix_cross_up = ema3[i] > 0 and ema3[i-1] <= 0
        trix_cross_down = ema3[i] < 0 and ema3[i-1] >= 0
        
        # Price relative to weekly trend
        price_above_weekly = close[i] > ema200_1w_aligned[i]
        price_below_weekly = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: TRIX crosses up, above weekly EMA200, with volume
            if trix_cross_up and price_above_weekly and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses down, below weekly EMA200, with volume
            elif trix_cross_down and price_below_weekly and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses down or price falls below weekly EMA200
            if trix_cross_down or not price_above_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses up or price rises above weekly EMA200
            if trix_cross_up or not price_below_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals