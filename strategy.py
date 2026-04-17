#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h trend filter (EMA50) and 1d volatility regime (ATR ratio) for entry timing.
Long when: price > 4h EMA50 (uptrend) AND 1d ATR(7)/ATR(30) > 1.5 (high volatility) AND RSI(14) < 30 (oversold bounce).
Short when: price < 4h EMA50 (downtrend) AND 1d ATR(7)/ATR(30) > 1.5 (high volatility) AND RSI(14) > 70 (overbought bounce).
Uses volatility expansion after compression for mean reversion in trending markets.
Session filter (08-20 UTC) reduces noise. Target 15-30 trades/year.
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR(7) and ATR(30)
    def calculate_atr(high_vals, low_vals, close_vals, window):
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period TR is just high-low
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_7_1d = calculate_atr(high_1d, low_1d, close_1d, 7)
    atr_30_1d = calculate_atr(high_1d, low_1d, close_1d, 30)
    
    # Avoid division by zero
    atr_ratio_1d = np.where(atr_30_1d > 0, atr_7_1d / atr_30_1d, 1.0)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 1h RSI(14) for entry timing
    def calculate_rsi(close_vals, window):
        delta = np.diff(close_vals, prepend=close_vals[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=window, adjust=False, min_periods=window).mean().values
        avg_loss = pd.Series(loss).ewm(span=window, adjust=False, min_periods=window).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14 = calculate_rsi(close, 14)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(rsi_14[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volatility expansion condition: ATR ratio > 1.5
        vol_expansion = atr_ratio_1d_aligned[i] > 1.5
        
        if position == 0:
            # Long: uptrend + volatility expansion + oversold
            if (close[i] > ema_50_4h_aligned[i] and 
                vol_expansion and 
                rsi_14[i] < 30):
                signals[i] = 0.20
                position = 1
            # Short: downtrend + volatility expansion + overbought
            elif (close[i] < ema_50_4h_aligned[i] and 
                  vol_expansion and 
                  rsi_14[i] > 70):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI reverts to midpoint or trend breaks
            if rsi_14[i] > 50 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI reverts to midpoint or trend breaks
            if rsi_14[i] < 50 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA50_1dATRratio_VolExpansion_RSI14_MeanReversion"
timeframe = "1h"
leverage = 1.0