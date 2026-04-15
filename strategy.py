#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d RSI mean reversion
# Designed for low trade frequency (target 20-40/year) with clear mean reversion logic
# Works in both bull (oversold bounce) and bear (overbought rejection) markets
# Uses Choppiness Index to identify ranging markets where mean reversion works best
# Entry: RSI(14) < 30 for long, > 70 for short when market is ranging (CHOP > 61.8)
# Exit: RSI returns to neutral zone (40-60) or trend emerges (CHOP < 38.2)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily timeframe
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on daily timeframe
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = np.maximum(high[1:], low[:-1]) - np.minimum(high[1:], low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # True Range sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(TR_sum / (HH - LL)) / log10(14)
    chop = 100 * np.log10(tr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # Align indicators to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]):
            continue
        
        # Long entry: RSI oversold + ranging market
        if (rsi_aligned[i] < 30 and 
            chop_aligned[i] > 61.8 and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: RSI overbought + ranging market
        elif (rsi_aligned[i] > 70 and 
              chop_aligned[i] > 61.8 and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral or trend emerges
        elif position == 1 and (rsi_aligned[i] > 40 or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned[i] < 60 or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_ChopRSI_MeanReversion"
timeframe = "4h"
leverage = 1.0