#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + RSI mean-reversion with chop filter
# Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction
# RSI(14) for mean-reversion entries when overextended against trend
# Choppiness Index filter to avoid ranging markets
# Targets 50-150 total trades over 4 years by combining trend-following with mean-reversion
# Works in bull (trend follow) and bear (mean revert in range) markets

name = "12h_kama_rsi_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA trend indicator (ER=10, Fast=2, Slow=30)
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series - close_series.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) for mean-reversion
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14-period) - regime filter
    atr1 = np.maximum(high - low, 
                      np.maximum(abs(high - np.roll(close, 1)), 
                                 abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Chop filter: only trade when market is trending (CHOP < 50)
            if chop[i] < 50:
                # Long when price above KAMA and RSI oversold
                if close[i] > kama[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Short when price below KAMA and RSI overbought
                elif close[i] < kama[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
            # In choppy markets, mean revert at extremes
            else:
                if rsi[i] < 20:  # deeply oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 80:  # deeply overbought
                    signals[i] = -0.25
                    position = -1
    
    return signals