#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_1dTrend_Volume_Confirm
# Hypothesis: TRIX (15) zero cross with 1-day EMA50 trend filter and volume confirmation.
# TRIX filters noise and detects momentum shifts. Long when TRIX crosses above zero and price above 1d EMA50,
# short when TRIX crosses below zero and price below 1d EMA50. Volume > 1.5x 20-period EMA confirms momentum.
# Works in bull/bear by aligning with higher timeframe trend. Targets 20-30 trades/year.

name = "4h_TRIX_ZeroCross_1dTrend_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Data (loaded ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === TRIX (15) on close ===
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=15, min_periods=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, min_periods=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, min_periods=15, adjust=False).mean()
    trix = ema3.pct_change() * 100  # percentage
    trix = trix.fillna(0).values
    
    # === 1d EMA50 Trend Filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers TRIX and EMA50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema50_1d_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + price above 1d EMA50 + volume spike
            if (trix[i-1] <= 0 and trix[i] > 0 and 
                close[i] > ema50_1d_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: TRIX crosses below zero + price below 1d EMA50 + volume spike
            elif (trix[i-1] >= 0 and trix[i] < 0 and 
                  close[i] < ema50_1d_4h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: TRIX crosses zero in opposite direction
            if position == 1:
                if trix[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if trix[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals