#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_RSI_MeanReversion_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for mean-reversion entries (long when RSI<30 in uptrend, short when RSI>70 in downtrend),
and Choppiness Index (CHOP) as regime filter to avoid whipsaws. Only trade when CHOP>50 (ranging market)
to capture reversals in range-bound conditions. Uses discrete sizing (0.25) to limit fee churn.
Designed for 1d timeframe with ~10-25 trades/year, works in both bull and bear markets by
fading extremes in ranging regimes while respecting higher-timeframe trend (1w) via KAMA slope.
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
    
    # 1w data for HTF trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI(14)
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
        avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index (CHOP)
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=length, min_periods=length).mean().values
        
        max_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        min_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        
        chop = 100 * np.log10(np.sum(atr, axis=1) / (max_high - min_low)) / np.log10(length)
        # Handle division by zero and invalid values
        chop = np.where((max_high - min_low) != 0, chop, 50)
        return chop
    
    # Pre-calculate indicators
    kama_vals = kama(close, length=10, fast=2, slow=30)
    rsi_vals = rsi(close, length=14)
    chop_vals = choppiness_index(high, low, close, length=14)
    
    # 1w EMA for trend filter (optional)
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(30, 21)  # KAMA, RSI, CHOP warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi_vals[i]) or 
            np.isnan(chop_vals[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: price above/below KAMA
        price_above_kama = close[i] > kama_vals[i]
        price_below_kama = close[i] < kama_vals[i]
        
        # Choppiness regime: only trade in ranging markets (CHOP > 50)
        ranging_market = chop_vals[i] > 50
        
        if position == 0:
            # Long: RSI oversold (<30) in ranging market
            long_signal = (rsi_vals[i] < 30) and ranging_market
            # Short: RSI overbought (>70) in ranging market
            short_signal = (rsi_vals[i] > 70) and ranging_market
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: RSI reverts to mean (>50) or chop drops below 40 (trending)
            if (rsi_vals[i] > 50) or (chop_vals[i] < 40):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: RSI reverts to mean (<50) or chop drops below 40 (trending)
            if (rsi_vals[i] < 50) or (chop_vals[i] < 40):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter_RSI_MeanReversion_ChopFilter"
timeframe = "1d"
leverage = 1.0