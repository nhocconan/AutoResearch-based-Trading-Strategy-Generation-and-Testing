#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend + RSI mean reversion + Chop regime filter
# Long when KAMA trending up, RSI < 40, and Chop > 61.8 (range)
# Short when KAMA trending down, RSI > 60, and Chop > 61.8 (range)
# Exit when RSI crosses 50 or Chop < 38.2 (trend)
# Uses KAMA for adaptive trend, RSI for mean reversion in range, Chop for regime
# Designed to capture mean reversion in ranging markets while avoiding trending whipsaws
# Target: 40-80 total trades over 4 years (10-20/year) with size 0.25

name = "1d_KAMA_RSI_Chop_MeanReversion"
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
    
    # Calculate 1w KAMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # KAMA calculation
    close_1w = df_1w['close']
    change = abs(close_1w.diff(10))
    volatility = close_1w.diff(1).abs().rolling(10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close_1w.iloc[0]]
    for i in range(1, len(close_1w)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1w.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_1w = kama
    
    # Align KAMA to daily
    kama_1d_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate daily Chop(14)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1))))
    atr_sum = atr.rolling(14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(14, min_periods=14).max()
    min_low = pd.Series(low).rolling(14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA up, RSI < 40, Chop > 61.8 (range)
            if (kama_1d_aligned[i] > kama_1d_aligned[i-1] and 
                rsi_values[i] < 40 and 
                chop_values[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down, RSI > 60, Chop > 61.8 (range)
            elif (kama_1d_aligned[i] < kama_1d_aligned[i-1] and 
                  rsi_values[i] > 60 and 
                  chop_values[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI >= 50 or Chop < 38.2 (trend)
            if (rsi_values[i] >= 50) or (chop_values[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI <= 50 or Chop < 38.2 (trend)
            if (rsi_values[i] <= 50) or (chop_values[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals