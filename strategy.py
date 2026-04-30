#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + chop regime filter.
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum confirmation,
# and Choppiness Index(14) to filter range-bound markets. Only takes longs when KAMA rising,
# RSI > 50, and CHOP < 38.2 (trending); shorts when KAMA falling, RSI < 50, and CHOP < 38.2.
# Discrete position sizing at ±0.25 to limit fee drag. Target: 30-100 total trades over 4 years (7-25/year).
# No session filter to allow global market participation.

name = "1d_KAMA_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute indicators with proper min_periods
    # KAMA: Efficiency Ratio over 10 periods, Fast=2, Slow=30
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.concatenate([np.full(14, np.nan), chop])
    
    # Align all indicators (though computed on 1d, we need to ensure no look-ahead)
    # Since we're on 1d timeframe, no alignment needed, but we'll use the same pattern for consistency
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for KAMA(10,2,30), RSI(14), CHOP(14)
    
    for i in range(start_idx, n):
        # Skip if any indicator is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama[i]
        curr_rsi = rsi[i]
        curr_chop = chop[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA rising (today > yesterday), RSI > 50, trending market (CHOP < 38.2)
            if (curr_close > curr_kama and  # price above KAMA indicates upward momentum
                curr_rsi > 50 and 
                curr_chop < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (price below KAMA), RSI < 50, trending market (CHOP < 38.2)
            elif (curr_close < curr_kama and  # price below KAMA indicates downward momentum
                  curr_rsi < 50 and 
                  curr_chop < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price crosses below KAMA or RSI < 40 or market becomes choppy
            if (curr_close < curr_kama or 
                curr_rsi < 40 or 
                curr_chop > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price crosses above KAMA or RSI > 60 or market becomes choppy
            if (curr_close > curr_kama or 
                curr_rsi > 60 or 
                curr_chop > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals