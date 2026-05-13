#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction + RSI(14) + choppiness regime filter on 1d timeframe.
# Long when KAMA is rising (trending up) AND RSI < 50 (mean reversion in uptrend) AND CHOP > 61.8 (range regime).
# Short when KAMA is falling (trending down) AND RSI > 50 (mean reversion in downtrend) AND CHOP > 61.8 (range regime).
# Uses ATR(14) trailing stop (2.5x) for risk control.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI extremes in the direction of the trend provide high-probability mean-reversion entries.
# Choppiness filter ensures we only trade in ranging markets where mean reversion works best.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d.

name = "1d_KAMA_RSI_Chop_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    net_change = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(net_change)):
        net_change[i] = np.abs(close[i] - close[i-10]) if i >= 10 else np.abs(close[i] - close[0])
    er = net_change / np.maximum(volatility, 1e-10)
    # Smoothing constants: fastest SC=2/(2+1)=0.67, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # CHOP = 100 * log10(sum(ATR14) / range14) / log10(14)
    chop = 100 * np.log10(np.divide(sum_atr_14, range_14, out=np.full_like(sum_atr_14, np.nan), where=range_14!=0)) / np.log10(14)
    
    # Align indicators to 1d timeframe (wait for 1d bar to close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (trending up) AND RSI < 50 (mean reversion in uptrend) AND CHOP > 61.8 (range regime)
            if kama_aligned[i] > kama_aligned[i-1] and rsi_aligned[i] < 50 and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: KAMA falling (trending down) AND RSI > 50 (mean reversion in downtrend) AND CHOP > 61.8 (range regime)
            elif kama_aligned[i] < kama_aligned[i-1] and rsi_aligned[i] > 50 and chop_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals