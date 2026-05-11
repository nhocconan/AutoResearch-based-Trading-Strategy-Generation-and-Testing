#!/usr/bin/env python3
"""
4h_Keltner_Channel_R1_S1_Breakout_1dTrend_Regime
Hypothesis: Use 1-day Keltner Channels (based on ATR) as dynamic support/resistance levels.
Breakout above upper band in uptrend (price > EMA50) goes long; breakdown below lower band in downtrend goes short.
Adds chop regime filter (Choppiness Index > 61.8) to avoid whipsaws in ranging markets.
Designed to work in both bull (buy breakouts) and bear (sell breakdowns) with trend filter.
Target: 20-40 trades per year on 4h timeframe.
"""

name = "4h_Keltner_Channel_R1_S1_Breakout_1dTrend_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1D Data for Keltner Channels and Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Keltner calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # True Range components for ATR
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - prev_close)
    tr3 = np.abs(prev_low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) for Keltner Channels
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channels: EMA20 ± 2*ATR
    ema_20 = pd.Series(prev_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Trend filter: EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Choppiness Index for regime filter (using daily data)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(prev_high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(prev_low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr14 / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_mask = chop > 61.8  # Chop > 61.8 = ranging market (avoid breakouts)
    
    # Align 1D indicators to 4h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_mask, additional_delay_bars=0)  # Boolean mask
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade when NOT in choppy regime (chop <= 61.8 = trending)
        in_chop = chop_aligned[i] if not np.isnan(chop_aligned[i]) else True
        
        if position == 0:
            # Long: price breaks above upper Keltner AND uptrend (price > EMA50) AND NOT choppy
            if close[i] > upper_keltner_aligned[i] and close[i] > ema_50_aligned[i] and not in_chop:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner AND downtrend (price < EMA50) AND NOT choppy
            elif close[i] < lower_keltner_aligned[i] and close[i] < ema_50_aligned[i] and not in_chop:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA50 OR re-enters Keltner channel
            if close[i] < ema_50_aligned[i] or close[i] < upper_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA50 OR re-enters Keltner channel
            if close[i] > ema_50_aligned[i] or close[i] > lower_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals