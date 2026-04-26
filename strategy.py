#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_RSI_Chop_Regime_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index (CHOP) for regime filtering.
Enter long when KAMA upward, RSI > 50, and CHOP < 38.2 (trending market).
Enter short when KAMA downward, RSI < 50, and CHOP < 38.2.
Exit when opposite signal or CHOP > 61.8 (choppy market) to avoid whipsaw.
Uses volume confirmation (volume > 1.5x 20-day median) to filter low-conviction moves.
Designed for low trade frequency (~10-25/year) with discrete position sizing (0.25) to minimize fee drag.
Works in both bull and bear markets by adapting to trending regimes via CHOP filter.
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
    
    # Get 1d data (primary) and 1w for HTF regime context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === PRIMARY INDICATORS ON 1D ===
    # KAMA (adaptive trend) - ER=10, FAST=2, SLOW=30
    close_1d = pd.Series(df_1d['close'].values)
    change = abs(close_1d.diff(10).values)
    volatility = abs(close_1d.diff(1)).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d.iloc[i] - kama[i-1])
    
    # RSI(14)
    delta = close_1d.diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP) - 14-period
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((hh - ll) > 0, chop, 50)  # default to neutral when range=0
    
    # Volume filter: volume > 1.5x 20-day median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_filter = volume > 1.5 * vol_median
    
    # === HTF REGIME FILTER FROM 1W (EMA50 trend) ===
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to lower timeframe (prices index)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.astype(float))
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA calculation (10), RSI (14), CHOP (14), volume median (20), EMA50 (50)
    start_idx = max(10, 14, 14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_filter_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        kama_val = kama_aligned[i]
        close_val = close[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_filter = volume_filter_aligned[i] > 0.5  # boolean
        ema_50w_val = ema_50_1w_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        choppy_exit = chop_val > 61.8  # exit when market becomes choppy
        
        # KAMA trend: price above/below KAMA
        kama_up = close_val > kama_val
        kama_down = close_val < kama_val
        
        # RSI momentum: bullish/bearish
        rsi_bullish = rsi_val > 50
        rsi_bearish = rsi_val < 50
        
        if position == 0:
            # Long: KAMA up, RSI > 50, trending regime, volume confirmation
            long_signal = kama_up and rsi_bullish and trending_regime and vol_filter
            
            # Short: KAMA down, RSI < 50, trending regime, volume confirmation
            short_signal = kama_down and rsi_bearish and trending_regime and vol_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA down OR RSI < 50 OR choppy regime
            if kama_down or (rsi_val < 50) or choppy_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA up OR RSI > 50 OR choppy regime
            if kama_up or (rsi_val > 50) or choppy_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0