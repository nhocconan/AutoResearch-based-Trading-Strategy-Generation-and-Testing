#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction + RSI(14) + chop filter (CHOP > 61.8) for mean reversion in ranging markets.
# Long when KAMA turns up, RSI < 40, and choppy regime (CHOP > 61.8).
# Short when KAMA turns down, RSI > 60, and choppy regime (CHOP > 61.8).
# Uses ATR(14) trailing stop (2.5x) for risk control.
# Uses discrete position sizing (0.25) to minimize fee churn.
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
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close, prepend=close[0]))
    # Correct volatility calculation: sum of absolute changes over ER period
    er_period = 10
    volatility_sum = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=er_period, min_periods=1).sum().values
    er = change / (volatility_sum + 1e-10)  # Avoid division by zero
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast=2, Slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) - using 14-period
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, period) - min(low, period))) / log10(period)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    
    # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
    chop_regime = chop > 61.8  # Only trade in ranging markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or 
            np.isnan(chop[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA turning up (current > previous), RSI < 40 (oversold), choppy regime
            if kama[i] > kama[i-1] and rsi[i] < 40 and chop_regime[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: KAMA turning down (current < previous), RSI > 60 (overbought), choppy regime
            elif kama[i] < kama[i-1] and rsi[i] > 60 and chop_regime[i]:
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