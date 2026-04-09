#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d KAMA trend + RSI + chop regime filter
# - Uses 1d HTF for KAMA(10) to identify trend direction
# - Entry when price closes above/below KAMA with RSI(14) confirmation and chop regime filter
# - Chop regime: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
# - ATR(14) trailing stop: exit at 2.5x ATR from extreme since entry
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: KAMA adapts to volatility, chop filter avoids whipsaws in ranging markets
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA (10-period)
    # Efficiency Ratio: ER = |Close - Close(10)| / Sum(|Close - Close(1)|) over 10 periods
    change_1d = np.abs(np.diff(close_1d, 10))  # |Close[t] - Close[t-10]|
    volatility_1d = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility_1d[i] = volatility_1d[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    volatility_1d[0:10] = np.sum(np.abs(np.diff(close_1d[0:11])) if len(close_1d) >= 11 else np.abs(np.diff(close_1d)))
    er = np.zeros_like(close_1d)
    er[10:] = change_1d[9:] / np.maximum(volatility_1d[10:], 1e-10)
    # Smoothing Constants: SC = [ER * (Fastest SC - Slowest SC) + Slowest SC]^2
    fastest_sc = 2 / (2 + 1)   # EMA(2)
    slowest_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    # KAMA: KAMA[t] = KAMA[t-1] + SC * (Price[t] - KAMA[t-1])
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (wait for completed 1d bar)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # Handle division by zero
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 12h Chop Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of True Range over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max - Min over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Chop Index = 100 * log10(sum_tr / range_14) / log10(14)
    chop = np.zeros_like(close)
    mask = (range_14 > 0) & (sum_tr > 0)
    chop[mask] = 100 * np.log10(sum_tr[mask] / range_14[mask]) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # Handle invalid values
    
    # ATR for stoploss
    atr = atr_12h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop[i]) or
            np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: KAMA trend + RSI confirmation + chop regime filter
            # Trending regime: CHOP < 38.2
            if chop[i] < 38.2:
                # Long entry: price above KAMA + RSI > 50 (bullish momentum)
                if close[i] > kama_aligned[i] and rsi_aligned[i] > 50:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price below KAMA + RSI < 50 (bearish momentum)
                elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals