# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index filter with 1d RSI mean reversion.
# Long when: 12h CHOP > 61.8 (range) AND 1d RSI < 30 (oversold)
# Short when: 12h CHOP > 61.8 (range) AND 1d RSI > 70 (overbought)
# Exit when: 12h CHOP < 38.2 (trending) OR RSI returns to neutral (40-60)
# This strategy targets range-bound markets where RSI mean reversion works.
# The Choppiness Index filter ensures we only trade in ranging conditions,
# avoiding whipsaws in strong trends. Works in both bull and bear markets
# as range-bound periods occur in all market regimes.

name = "12h_CHOP_RSI_Range"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 12h Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * (np.log10(atr_sum) - np.log10(max_high - min_low)) / np.log10(14)
    chop = chop.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 14)  # Sufficient warmup for CHOP and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(chop[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ranging market + RSI oversold
            long_cond = (chop[i] > 61.8) and (rsi_aligned[i] < 30)
            # Short: ranging market + RSI overbought
            short_cond = (chop[i] > 61.8) and (rsi_aligned[i] > 70)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trending market OR RSI returns to neutral
            if chop[i] < 38.2 or (rsi_aligned[i] >= 40 and rsi_aligned[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trending market OR RSI returns to neutral
            if chop[i] < 38.2 or (rsi_aligned[i] >= 40 and rsi_aligned[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals