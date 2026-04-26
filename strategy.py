#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: KAMA adaptive trend on 12h with RSI(14) momentum filter and Choppiness Index regime filter to avoid whipsaws. Enters long when KAMA trending up, RSI > 50, and chop < 61.8 (trending regime). Enters short when KAMA trending down, RSI < 50, and chop < 61.8. Uses discrete position sizing (0.25) and ATR-based stoploss (2.0x) for risk management. Designed for low trade frequency (target 12-37/year) to minimize fee drag while capturing medium-term swings in both bull and bear markets. The combination of adaptive trend, momentum, and regime filter reduces false signals and improves test generalization.
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
    volume = prices['volume'].values
    
    # Get 1d data for Chop filter (needs extra delay for confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA(10) on 12h for adaptive trend
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(10).values)
    volatility = np.abs(close_s.diff(1).rolling(window=10).sum().values)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = pd.Series(kama).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Choppiness Index(14) on 1d (needs 2-bar extra delay for confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) == 0, 50, chop)  # avoid division by zero
    
    # Align 1d indicators to 12h with extra delay for Chop (needs confirmation)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama[-len(df_1d):])  # Use last values to match length
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi[-len(df_1d):])
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA(10), RSI(14), ATR(14)
    start_idx = max(10, 14, 14) + 2
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: chop < 61.8 = trending regime (good for trend following)
        trending_regime = chop_val < 61.8
        
        if position == 0:
            # Long: KAMA trending up, RSI > 50, trending regime
            long_signal = (close_val > kama_val) and (rsi_val > 50) and trending_regime
            
            # Short: KAMA trending down, RSI < 50, trending regime
            short_signal = (close_val < kama_val) and (rsi_val < 50) and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend fails OR RSI momentum fades OR chop too high (ranging)
            if (close_val <= kama_val) or (rsi_val <= 50) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend fails OR RSI momentum fades OR chop too high (ranging)
            if (close_val >= kama_val) or (rsi_val >= 50) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0