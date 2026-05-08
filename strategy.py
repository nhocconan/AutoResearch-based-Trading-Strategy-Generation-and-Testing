#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d RSI mean reversion.
# In ranging markets (CHOP > 61.8), enter long when 1d RSI < 30 and short when RSI > 70.
# Exit when RSI crosses back above 50 (long) or below 50 (short).
# Choppiness Index filters out trending markets where mean reversion fails.
# RSI provides mean-reversion signals in ranging regimes.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Chop_RSI_MeanReversion_1d"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period RSI on daily close
    delta = pd.Series(df_1d['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle no loss case
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 14-period Choppiness Index on 12h data
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 
                    100 * np.log10(np.sum(atr) / range_hl) / np.log10(14), 
                    50)
    
    # Choppiness regime: > 61.8 = ranging (good for mean reversion)
    chop_ranging = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Sufficient warmup for RSI and CHOP
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(chop_ranging[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in ranging markets
            if chop_ranging[i]:
                # Long when RSI oversold (< 30)
                if rsi_aligned[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Short when RSI overbought (> 70)
                elif rsi_aligned[i] > 70:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 50
            if rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses back below 50
            if rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals