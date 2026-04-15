#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter with 1w RSI mean reversion
# In high choppiness (range) markets, fade extreme RSI on weekly timeframe
# In low choppiness (trending) markets, follow weekly RSI momentum
# Designed for low trade frequency (10-25/year) with clear regime adaptation
# Works in bull markets (trend following) and bear markets (mean reversion in ranges)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data (primary timeframe) for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1d Choppiness Index (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Calculate 1w RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 1d timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i]):
            continue
        
        chop_val = chop_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Regime-based logic
        if chop_val > 61.8:  # High chop = range market -> mean reversion
            # Sell when RSI overbought, buy when RSI oversold
            if rsi_val > 70 and position >= 0:  # Overbought -> short
                position = -1
                signals[i] = -base_size
            elif rsi_val < 30 and position <= 0:  # Oversold -> long
                position = 1
                signals[i] = base_size
            # Exit when RSI returns to neutral zone
            elif position == 1 and rsi_val >= 50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_val <= 50:
                position = 0
                signals[i] = 0.0
        else:  # Low chop = trending market -> follow momentum
            # Buy when RSI > 50, sell when RSI < 50
            if rsi_val > 50 and position <= 0:
                position = 1
                signals[i] = base_size
            elif rsi_val < 50 and position >= 0:
                position = -1
                signals[i] = -base_size
            # Exit on reverse signal
            elif position == 1 and rsi_val < 50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_val > 50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_ChopRSI_Regime_MeanRev"
timeframe = "1d"
leverage = 1.0