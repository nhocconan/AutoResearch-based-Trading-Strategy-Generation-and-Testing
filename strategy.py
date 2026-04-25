#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_And_Chop_Filter
Hypothesis: 4h KAMA trend direction combined with RSI extremes and choppiness regime filter.
Long when KAMA trending up (price > KAMA) AND RSI < 30 (oversold) AND choppy market (CHOP > 61.8).
Short when KAMA trending down (price < KAMA) AND RSI > 70 (overbought) AND choppy market (CHOP > 61.8).
Exit via ATR trailing stop (2.0*ATR from extreme) or opposite signal.
Designed for ~20-50 trades/year by requiring confluence of trend, mean reversion, and regime filters.
Works in bull/bear markets: KAMA captures trend, RSI catches pullbacks in trends, chop filter avoids whipsaws in ranging markets.
"""

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
    volume = prices['volume'].values
    
    # KAMA calculation (ER=10, Fast=2, Slow=30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) >= 11 else np.zeros(len(close)-10)
    # Pad arrays to match length
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index (CHOP) - 14 period
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = np.where(atr_sum > 0, 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(atr_period), 50)
    chop = np.concatenate([np.full(atr_period-1, np.nan), chop])
    
    # ATR for trailing stop
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 30)  # KAMA needs 10, RSI 14, CHOP 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: KAMA uptrend + RSI oversold + choppy market
            long_signal = (close[i] > kama[i]) and (rsi[i] < 30) and (chop[i] > 61.8)
            # Short: KAMA downtrend + RSI overbought + choppy market
            short_signal = (close[i] < kama[i]) and (rsi[i] > 70) and (chop[i] > 61.8)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: ATR trailing stop OR opposite signal
            atr_stop = long_extreme - 2.0 * atr[i]
            opposite_signal = (close[i] < kama[i]) and (rsi[i] > 70) and (chop[i] > 61.8)
            if close[i] <= atr_stop or opposite_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions: ATR trailing stop OR opposite signal
            atr_stop = short_extreme + 2.0 * atr[i]
            opposite_signal = (close[i] > kama[i]) and (rsi[i] < 30) and (chop[i] > 61.8)
            if close[i] >= atr_stop or opposite_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_With_RSI_And_Chop_Filter"
timeframe = "4h"
leverage = 1.0