#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA21 pullback with 4h trend filter and volume confirmation.
# Uses 4h EMA50 for trend direction, 1h EMA21 for entry timing, and volume spike for confirmation.
# Works in both bull and bear by only taking long in 4h uptrend and short in 4h downtrend.
# Target: 30-60 trades/year per symbol to avoid excessive fee drag.
name = "1h_EMA21_Pullback_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h trend filter: 50-period EMA on close
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMA21 for pullback entry
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike detection (20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema > 0, volume / vol_ema, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA and trend
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_21[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: pullback to EMA21 in uptrend with volume spike
            long_condition = (close[i] <= ema_21[i]) and uptrend and vol_spike[i]
            # Short: pullback to EMA21 in downtrend with volume spike
            short_condition = (close[i] >= ema_21[i]) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price crosses above EMA21 or trend turns down
            if (close[i] > ema_21[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price crosses below EMA21 or trend turns up
            if (close[i] < ema_21[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals