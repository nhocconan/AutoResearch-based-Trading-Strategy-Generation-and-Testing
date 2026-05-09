#!/usr/bin/env python3
# Hypothesis: 4h KAMA + RSI + Chop regime for regime-adaptive trading
# KAMA identifies trend direction, RSI(14) identifies momentum extremes
# Chop(14) determines market regime: Chop > 61.8 = range (mean revert), Chop < 38.2 = trending (trend follow)
# Long: Chop > 61.8 AND RSI < 30 AND price > KAMA (mean reversion in range)
# Short: Chop > 61.8 AND RSI > 70 AND price < KAMA (mean reversion in range)
# Long: Chop < 38.2 AND price > KAMA (trend follow)
# Short: Chop < 38.2 AND price < KAMA (trend follow)
# Exit: Opposite condition or Chop regime shift
# Position size: 0.25 to limit drawdown. Target: 20-40 trades/year.

name = "4h_KAMA_RSI_Chop_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Chop(14) - Choppy Index
    atr = pd.Series(high - low).rolling(window=14, min_periods=14).mean()
    true_range = np.maximum(high - low, np.maximum(abs(high - close_s.shift(1)), abs(low - close_s.shift(1))))
    atr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Range market: Chop > 61.8 -> mean reversion at RSI extremes
            if chop[i] > 61.8:
                if rsi[i] < 30 and close[i] > kama[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and close[i] < kama[i]:
                    signals[i] = -0.25
                    position = -1
            # Trending market: Chop < 38.2 -> follow trend
            elif chop[i] < 38.2:
                if close[i] > kama[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: opposite condition or regime shift
            if (chop[i] > 61.8 and rsi[i] > 50) or (chop[i] < 38.2 and close[i] < kama[i]) or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: opposite condition or regime shift
            if (chop[i] > 61.8 and rsi[i] < 50) or (chop[i] < 38.2 and close[i] > kama[i]) or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals