#!/usr/bin/env python3
# 1d_WeeklyKeltner_Retest_Strategy
# Hypothesis: Weekly Keltner Channel mid-line (EMA20) acts as dynamic support/resistance.
# Price retests to the EMA20 with volume confirmation provide high-probability entries in both bull and bear markets.
# Uses 1w EMA20 as the primary trend filter and 1d price action for entry timing.
# Low-frequency design targets 15-30 trades/year to minimize fee drag.

name = "1d_WeeklyKeltner_Retest_Strategy"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 (Keltner mid-line)
    weekly_close = df_1w['close'].values
    ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to daily timeframe (waits for weekly close)
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20)
    
    # Calculate daily ATR for volatility filter and stop reference
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price retests to weekly EMA20 from above + volume
            if (low[i] <= ema20_aligned[i] * 1.005 and  # Within 0.5% above EMA
                close[i] > ema20_aligned[i] and         # Close above EMA
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price retests to weekly EMA20 from below + volume
            elif (high[i] >= ema20_aligned[i] * 0.995 and  # Within 0.5% below EMA
                  close[i] < ema20_aligned[i] and          # Close below EMA
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly EMA20
            if close[i] < ema20_aligned[i] * 0.995:  # 0.5% below EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly EMA20
            if close[i] > ema20_aligned[i] * 1.005:  # 0.5% above EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals