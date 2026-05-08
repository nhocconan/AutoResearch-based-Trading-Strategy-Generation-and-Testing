#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Wick_Reversal_With_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1d Wick Reversal Signals ===
    # Bullish reversal: long lower wick > 2x body and close > open
    body_1d = np.abs(close_1d - df_1d['open'].values)
    lower_wick_1d = np.minimum(close_1d, df_1d['open'].values) - df_1d['low'].values
    bullish_wick = (lower_wick_1d > 2 * body_1d) & (close_1d > df_1d['open'].values)
    
    # Bearish reversal: long upper wick > 2x body and close < open
    upper_wick_1d = df_1d['high'].values - np.maximum(close_1d, df_1d['open'].values)
    bearish_wick = (upper_wick_1d > 2 * body_1d) & (close_1d < df_1d['open'].values)
    
    # Align wick signals to 6h timeframe
    bullish_wick_aligned = align_htf_to_ltf(prices, df_1d, bullish_wick.astype(float))
    bearish_wick_aligned = align_htf_to_ltf(prices, df_1d, bearish_wick.astype(float))
    
    # === 6h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bullish_wick_aligned[i]) or 
            np.isnan(bearish_wick_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long on bullish wick reversal in uptrend (price > EMA50)
            long_cond = bullish_wick_aligned[i] and (close[i] > ema50_1d_aligned[i]) and (volume[i] > vol_ma20[i])
            
            # Enter short on bearish wick reversal in downtrend (price < EMA50)
            short_cond = bearish_wick_aligned[i] and (close[i] < ema50_1d_aligned[i]) and (volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long on bearish wick reversal or trend break
            exit_cond = bearish_wick_aligned[i] or (close[i] < ema50_1d_aligned[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on bullish wick reversal or trend break
            exit_cond = bullish_wick_aligned[i] or (close[i] > ema50_1d_aligned[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Wick reversal strategy on 1d timeframe with trend filter and volume confirmation.
# Enters long on bullish 1d wick rejection when 6h price is above 1d EMA50 (uptrend),
# enters short on bearish 1d wick rejection when 6h price is below 1d EMA50 (downtrend).
# Uses volume confirmation to ensure institutional participation. Designed to work in
# both bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag.
# Uses discrete sizing (0.25) to reduce churn. Works on BTC/ETH via institutional
# rejection levels on higher timeframe.