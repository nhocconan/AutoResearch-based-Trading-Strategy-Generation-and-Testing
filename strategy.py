#!/usr/bin/env python3
# 6H_200EMA_Cross_Signal
# Hypothesis: Price crossing above/below 200-period EMA on 6h chart indicates regime change.
# Long when: price crosses above 200 EMA with bullish volume confirmation.
# Short when: price crosses below 200 EMA with bearish volume confirmation.
# Uses 12h trend filter: only take signals aligned with 12h EMA50 trend.
# Works in bull/bear by following long-term trend and filtering with volume.
# Target: 15-30 trades/year per symbol.

name = "6H_200EMA_Cross_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # EMA200 for trend detection
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # EMA50 for short-term trend
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    bullish_12h = close_12h > ema50_12h
    bearish_12h = close_12h < ema50_12h
    
    # Align 12h trend to 6h
    bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_12h.astype(float))
    bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema200[i]) or np.isnan(ema50[i]) or np.isnan(vol_ma[i]) or
            np.isnan(bullish_12h_aligned[i]) or np.isnan(bearish_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.3
        
        bullish_12h_aligned_val = bullish_12h_aligned[i] > 0.5
        bearish_12h_aligned_val = bearish_12h_aligned[i] > 0.5
        
        # Detect EMA200 crossovers
        if i > start_idx:
            ema200_prev = ema200[i-1]
            ema200_curr = ema200[i]
            close_prev = close[i-1]
            close_curr = close[i]
            
            # Bullish crossover: price crosses above EMA200
            bullish_cross = (close_prev <= ema200_prev) and (close_curr > ema200_curr)
            # Bearish crossover: price crosses below EMA200
            bearish_cross = (close_prev >= ema200_prev) and (close_curr < ema200_curr)
            
            if position == 0:
                # Enter long: bullish cross + bullish 12h trend + volume confirmation
                if bullish_cross and bullish_12h_aligned_val and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Enter short: bearish cross + bearish 12h trend + volume confirmation
                elif bearish_cross and bearish_12h_aligned_val and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Exit long: bearish cross or trend reversal
                if bearish_cross or not bullish_12h_aligned_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Exit short: bullish cross or trend reversal
                if bullish_cross or not bearish_12h_aligned_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals