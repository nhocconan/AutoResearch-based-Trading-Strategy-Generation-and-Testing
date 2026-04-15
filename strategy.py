#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI + chop filter
# Uses KAMA(14) to determine trend direction, RSI(14) for overbought/oversold conditions,
# and Choppiness Index(14) to filter ranging markets. Only trades when trend is aligned
# with RSI extremes and market is trending (CHOP < 38.2). Designed to work in both bull
# and bear markets by following KAMA direction and avoiding false signals in ranging markets.
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA, RSI, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (14-period) on 1d
    # Efficiency Ratio
    change = np.abs(close_1d - np.roll(close_1d, 14))
    change[0:14] = np.nan
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    
    # Proper volatility calculation (sum of absolute changes over 14 periods)
    volatility = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-14:i+1])))
    volatility[0:14] = np.nan
    
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # using fast=2, slow=30 as typical
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period) on 1d
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Choppiness Index
    chop = np.where(range_max_min != 0, -100 * np.log10(tr_sum / range_max_min) / np.log10(14), 50)
    
    # Align indicators to lower timeframe (assuming we're working on 1d timeframe, so no alignment needed)
    # But we'll keep the structure for consistency with the requirement
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Long entry: price > KAMA + RSI < 30 (oversold) + CHOP < 38.2 (trending)
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] < 30 and
            chop_aligned[i] < 38.2):
            signals[i] = base_size
        
        # Short entry: price < KAMA + RSI > 70 (overbought) + CHOP < 38.2 (trending)
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] > 70 and
              chop_aligned[i] < 38.2):
            signals[i] = -base_size
        
        # Exit: opposite condition or chop > 61.8 (ranging market)
        elif ((close[i] > kama_aligned[i] and rsi_aligned[i] > 70) or
              (close[i] < kama_aligned[i] and rsi_aligned[i] < 30) or
              chop_aligned[i] > 61.8):
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0