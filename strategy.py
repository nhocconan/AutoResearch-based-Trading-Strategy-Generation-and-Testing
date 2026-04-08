#!/usr/bin/env python3
# 6h_1d_volatility_mean_reversion_v1
# Hypothesis: In 6h timeframe, mean-reversion opportunities arise when volatility spikes
# (ATR ratio > 2.0) and price reaches Bollinger Bands extremes (2.5 std) in the opposite
# direction of the 1d trend. Uses 1d EMA50 as trend filter: long only when price > EMA50,
# short only when price < EMA50. Exit when volatility contracts (ATR ratio < 1.2) or
# price returns to mean (middle Bollinger Band). Designed for 12-30 trades/year to
# minimize fee drag while capturing volatility mean reversion in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_volatility_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2.5) on 6h
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.5 * bb_std
    bb_lower = bb_middle - 2.5 * bb_std
    
    # ATR for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR(7) / ATR(30) to detect volatility spikes
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = np.where(atr30 > 0, atr7 / atr30, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: volatility contracts OR price returns to mean
            if atr_ratio[i] < 1.2 or close[i] > bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: volatility contracts OR price returns to mean
            if atr_ratio[i] < 1.2 or close[i] < bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volatility spike condition
            vol_spike = atr_ratio[i] > 2.0
            
            # Long entry: volatility spike + price at lower BB + uptrend (price > EMA50)
            if vol_spike and close[i] <= bb_lower[i] and close[i] > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: volatility spike + price at upper BB + downtrend (price < EMA50)
            elif vol_spike and close[i] >= bb_upper[i] and close[i] < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals