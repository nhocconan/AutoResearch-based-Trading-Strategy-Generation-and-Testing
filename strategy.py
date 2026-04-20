#!/usr/bin/env python3
# 1d_1w_Keltner_MR_Reversal_V1
# Hypothesis: On daily timeframe, price mean-reverts from Keltner Channel extremes during low-volatility regimes.
# In ranging markets (weekly ATR < 20-day ATR), price tends to revert from upper/lower Keltner bands.
# In high volatility (weekly ATR >= 20-day ATR), trend continues. Uses volume confirmation for reversals.
# Targets 10-25 trades/year by requiring volatility regime + band touch + volume spike.
# Works in bull/bear markets via volatility regime filter.

name = "1d_1w_Keltner_MR_Reversal_V1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-day ATR for volatility regime
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly ATR (using weekly high/low/close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1w = high_1w[1:] - low_1w[1:]
    tr2w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1w, np.maximum(tr2w, tr3w))])
    
    atr_1w = pd.Series(tr_w).rolling(window=6, min_periods=6).mean().values  # 6 weeks ~ 1.5 months
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate Keltner Channel (20-day EMA +/- 2*ATR)
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    upper_keltner = ema_20 + 2 * atr_20
    lower_keltner = ema_20 - 2 * atr_20
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Low volatility regime: weekly ATR < 2x daily ATR (range-bound market)
            if atr_1w_aligned[i] < 2 * atr_20[i]:
                # Mean reversion from lower band with volume confirmation
                if (close[i] <= lower_keltner[i] * 1.002 and 
                    volume[i] > 1.8 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Mean reversion from upper band with volume confirmation
                elif (close[i] >= upper_keltner[i] * 0.998 and 
                      volume[i] > 1.8 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
            # High volatility regime: trend continuation (no mean reversion)
            # Optional: could add trend-following logic here if desired
        
        elif position == 1:
            # Long exit: return to mean (EMA) or volatility regime shifts
            if (close[i] >= ema_20[i] * 0.998) or (atr_1w_aligned[i] >= 2 * atr_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to mean (EMA) or volatility regime shifts
            if (close[i] <= ema_20[i] * 1.002) or (atr_1w_aligned[i] >= 2 * atr_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals