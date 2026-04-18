#!/usr/bin/env python3
"""
1h EMA Crossover with 4h Trend Filter and Volume Spike
Hypothesis: During strong 4h trends (EMA50 > EMA200), 1-hour EMA crossovers (9/21) with volume spikes (1.5x average) capture momentum with controlled frequency. Works in bull/bear by following 4h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h indicators
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    ema40_4h = pd.Series(df_4h['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema40_4h_aligned = align_htf_to_ltf(prices, df_4h, ema40_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup for 4h EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema40_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        bullish_trend = ema40_4h_aligned[i] > ema200_4h_aligned[i]
        bearish_trend = ema40_4h_aligned[i] < ema200_4h_aligned[i]
        
        # Volume spike
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish 4h trend + EMA9 > EMA21 + volume spike
            if bullish_trend and ema9[i] > ema21[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: bearish 4h trend + EMA9 < EMA21 + volume spike
            elif bearish_trend and ema9[i] < ema21[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: EMA9 < EMA21 or trend change
            if ema9[i] < ema21[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: EMA9 > EMA21 or trend change
            if ema9[i] > ema21[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA_Crossover_4Trend_Volume"
timeframe = "1h"
leverage = 1.0