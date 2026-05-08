#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d RSI mean reversion
# Long when CHOP(14) > 61.8 (range) and RSI(14) < 30 (oversold)
# Short when CHOP(14) > 61.8 (range) and RSI(14) > 70 (overbought)
# Exit when CHOP(14) < 38.2 (trending) or RSI reverts to neutral (40-60)
# Uses 1d RSI for mean reversion signals and 12h Choppiness Index for regime filtering
# Works in both bull and bear markets by only taking mean-reversion trades in ranging markets
# Targets 12-37 trades per year (50-150 over 4 years) for low fee drag

name = "12h_Chop618_RSI14_MeanRev"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index on 12h data
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    chop = 100 * np.log10((atr * atr_period) / (highest_high - lowest_low)) / np.log10(atr_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        rsi_val = rsi_1d_aligned[i]
        
        if position == 0:
            # Enter long: ranging market (CHOP > 61.8) and oversold (RSI < 30)
            if chop_val > 61.8 and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Enter short: ranging market (CHOP > 61.8) and overbought (RSI > 70)
            elif chop_val > 61.8 and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trending market (CHOP < 38.2) or RSI returns to neutral (> 40)
            if chop_val < 38.2 or rsi_val > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trending market (CHOP < 38.2) or RSI returns to neutral (< 60)
            if chop_val < 38.2 or rsi_val < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals