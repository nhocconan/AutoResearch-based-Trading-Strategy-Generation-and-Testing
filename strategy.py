#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to volatility regime, RSI identifies overbought/oversold,
# Chop filter avoids whipsaws in ranging markets. Works in bull via KAMA uptrend + RSI pullback,
# in bear via KAMA downtrend + RSI bounce. Chop filter reduces false signals during consolidation.
# Target: 15-25 trades/year to minimize fee drag.
name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Chop filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average)
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chop (weekly)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high[1:] - weekly_low[1:]
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(weekly_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(weekly_low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Align Chop to daily
    chop_aligned = align_htf_to_ltf(prices, df_weekly, chop)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi.iloc[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: Chop > 50 indicates ranging market (mean reversion favorable)
        chop_filter = chop_aligned[i] > 50
        
        if position == 1:  # Long position
            # Exit: RSI overbought or KAMA turns down
            if rsi.iloc[i] > 70 or close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: RSI oversold or KAMA turns up
            if rsi.iloc[i] < 30 or close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price above KAMA + RSI oversold + chop filter
            if close[i] > kama[i] and rsi.iloc[i] < 35 and chop_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price below KAMA + RSI overbought + chop filter
            elif close[i] < kama[i] and rsi.iloc[i] > 65 and chop_filter:
                position = -1
                signals[i] = -0.25
    
    return signals