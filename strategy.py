#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend + RSI momentum + chop regime filter for 1d timeframe.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI(14) confirms momentum direction. Choppiness Index > 61.8 filters range-bound conditions.
# Designed for 1d to capture multi-day trends with low frequency, suitable for bull and bear markets.
# Entry: Long when KAMA rising, RSI > 50, and CHOP > 61.8 (range) for mean-reversion longs at support.
#        Short when KAMA falling, RSI < 50, and CHOP > 61.8 for mean-reversion shorts at resistance.
# Exit: Opposite KAMA direction or RSI reversal.
# Uses strict conditions to limit trades (~10-20/year) and avoid overtrading.
name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA ( Kaufman Adaptive Moving Average ) - 10-period ER, 2/30 SC
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [np.nan] * len(close)
    kama[9] = close_s.iloc[9]  # seed
    for i in range(10, len(close)):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama = np.array(kama)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Choppiness Index (14-period)
    atr = []
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    atr_s = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    atr = atr_s.values
    high_roll = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_roll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_s * 14 / (high_roll - low_roll)) / np.log10(2)
    chop = np.nan_to_num(chop, nan=50.0)  # default to neutral if undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            i == 0 or np.isnan(kama[i-1])):
            signals[i] = 0.0
            continue
        
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA rising, bullish momentum, in chop (range) for mean-reversion at support
            if kama_rising and rsi[i] > 50 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, bearish momentum, in chop (range) for mean-reversion at resistance
            elif kama_falling and rsi[i] < 50 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if KAMA falls or RSI turns bearish
            if not kama_rising or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if KAMA rises or RSI turns bullish
            if not kama_falling or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals