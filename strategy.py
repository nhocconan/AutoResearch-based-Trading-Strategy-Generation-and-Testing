#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator uses three SMAs (Jaw: 13, Teeth: 8, Lips: 5) to identify trends.
# Long when Lips > Teeth > Jaw (bullish alignment) in uptrend, short when Lips < Teeth < Jaw (bearish alignment) in downtrend.
# Uses 1d EMA34 trend filter and volume spike confirmation to filter entries.
# Designed to work in both bull and bear markets by following the 1d trend direction.
# Target: 20-30 trades/year per symbol to avoid excessive fee drag.
name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (SMAs)
    # Jaw: 13-period SMA (slow)
    sma_jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMA (medium)
    sma_teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMA (fast)
    sma_lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # 6h volume average for spike detection
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for SMA calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(sma_jaw[i]) or 
            np.isnan(sma_teeth[i]) or np.isnan(sma_lips[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Alligator alignment
        bullish_alignment = (sma_lips[i] > sma_teeth[i]) and (sma_teeth[i] > sma_jaw[i])
        bearish_alignment = (sma_lips[i] < sma_teeth[i]) and (sma_teeth[i] < sma_jaw[i])
        
        if position == 0:
            # Long condition: bullish alignment, in uptrend with volume spike
            long_condition = bullish_alignment and vol_spike[i] and uptrend
            # Short condition: bearish alignment, in downtrend with volume spike
            short_condition = bearish_alignment and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish alignment forms or trend turns down
            if bearish_alignment or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish alignment forms or trend turns up
            if bullish_alignment or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals