#!/usr/bin/env python3
# 1d_Weekly_MACD_Signal_Crossover
# Hypothesis: Weekly MACD crossovers on the 1-day chart capture major trend changes.
# Using 12,26,9 EMA parameters with confirmation from weekly trend (EMA50) and volume.
# Works in bull markets via bullish crossovers above weekly EMA50, and in bear markets via
# bearish crossovers below weekly EMA50. Volume filter ensures institutional participation.
# Target: 15-25 trades per year (~60-100 over 4 years) with position size 0.25.

name = "1d_Weekly_MACD_Signal_Crossover"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE for MACD and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Calculate weekly MACD (12,26,9)
    close_1w = df_1w['close'].values
    ema12 = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align weekly MACD components to daily timeframe
    macd_line_aligned = align_htf_to_ltf(prices, df_1w, macd_line)
    signal_line_aligned = align_htf_to_ltf(prices, df_1w, signal_line)
    macd_hist_aligned = align_htf_to_ltf(prices, df_1w, macd_hist)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(macd_line_aligned[i]) or np.isnan(signal_line_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # MACD crossover conditions
        macd_bullish_cross = macd_line_aligned[i] > signal_line_aligned[i] and macd_line_aligned[i-1] <= signal_line_aligned[i-1]
        macd_bearish_cross = macd_line_aligned[i] < signal_line_aligned[i] and macd_line_aligned[i-1] >= signal_line_aligned[i-1]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        # Trend filter from weekly EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: bullish MACD crossover + volume + uptrend
            if macd_bullish_cross and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish MACD crossover + volume + downtrend
            elif macd_bearish_cross and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish MACD crossover or trend reversal
            if macd_bearish_cross or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish MACD crossover or trend reversal
            if macd_bullish_cross or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals