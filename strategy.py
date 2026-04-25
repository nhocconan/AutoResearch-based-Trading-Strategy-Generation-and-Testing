#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v2
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend,
RSI(14) for momentum strength, and Choppiness Index for regime filter.
- Long when: price > KAMA(10,2,30), RSI(14) > 50, and CHOP(14) < 61.8 (trending regime)
- Short when: price < KAMA(10,2,30), RSI(14) < 50, and CHOP(14) < 61.8 (trending regime)
- Exit when trend reverses or market becomes choppy (CHOP > 61.8)
- Position size: 0.25. Target: 30-100 trades over 4 years (7-25/year).
- Works in bull/bear: KAMA adapts to volatility, RSI confirms momentum, CHOP filters ranging markets.
"""

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
    
    # Get 1d data for HTF (though primary is 1d, we use 1d for HTF calculations)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA(10,2,30) on 1d
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Handle edge cases for volatility
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.divide(change, volatility, out=np.full_like(change, np.nan, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # (ER * (fastest - slowest) + slowest)^2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Use Wilder's smoothing (alpha = 1/14)
    avg_gain = np.full_like(close_1d, np.nan, dtype=float)
    avg_loss = np.full_like(close_1d, np.nan, dtype=float)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan, dtype=float), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index(14) on 1d
    # True Range
    tr1 = np.abs(np.subtract(high_1d[1:], low_1d[:-1]))
    tr2 = np.abs(np.subtract(high_1d[1:], close_1d[:-1]))
    tr3 = np.abs(np.subtract(low_1d[1:], close_1d[:-1]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    # ATR(14)
    atr = np.full_like(close_1d, np.nan, dtype=float)
    atr[13] = np.mean(tr[1:15])  # seed
    for i in range(14, len(close_1d)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # Sum of ATR over 14 periods
    sum_atr = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(13, len(close_1d)):
        if i >= 13:
            sum_atr[i] = np.sum(atr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    max_high = np.full_like(close_1d, np.nan, dtype=float)
    min_low = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(13, len(close_1d)):
        max_high[i] = np.max(high_1d[i-13:i+1])
        min_low[i] = np.min(low_1d[i-13:i+1])
    # Chop = 100 * log10(sum(ATR) / (maxHigh - minLow)) / log10(14)
    range_hl = max_high - min_low
    chop = np.full_like(close_1d, np.nan, dtype=float)
    mask = (range_hl > 0) & (~np.isnan(sum_atr))
    chop[mask] = 100 * np.log10(sum_atr[mask] / range_hl[mask]) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA(10,2,30), RSI(14), CHOP(14)
    start_idx = 30  # KAMA needs ~30 for stability
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend and regime
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        trending_regime = chop_aligned[i] < 61.8  # trending when CHOP < 61.8
        choppy_regime = chop_aligned[i] > 61.8    # choppy when CHOP > 61.8
        
        if position == 0:
            # Enter long in bullish trending market
            long_setup = price_above_kama and rsi_bullish and trending_regime
            # Enter short in bearish trending market
            short_setup = price_below_kama and rsi_bearish and trending_regime
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal or choppy market
            exit_signal = (not price_above_kama) or (not rsi_bullish) or choppy_regime
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or choppy market
            exit_signal = (not price_below_kama) or (not rsi_bearish) or choppy_regime
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0