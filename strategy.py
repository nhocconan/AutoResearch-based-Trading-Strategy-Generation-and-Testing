#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Regime_Filter_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h for trend direction, combined with 1w choppiness index regime filter to avoid whipsaws in ranging markets. Enter long when price > KAMA and market is trending (CHOP < 38.2), short when price < KAMA and CHOP < 38.2. Exit when opposite condition occurs. Uses ATR-based trailing stop (2.0x) for risk management. Designed for low trade frequency (<30/year) by requiring strong trend regime confirmation. Works in bull markets (trend following) and bear markets (avoids false signals via regime filter). Focus on BTC/ETH as primary targets.
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
    
    # Get 1w data for HTF choppiness index regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Choppiness Index (14-period)
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros(len(high))
        atr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = (atr[i-1] * (period-1) + tr) / period
        
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        chop = np.zeros(len(high))
        for i in range(period-1, len(high)):
            if atr[i] > 0 and highest_high[i] > lowest_low[i]:
                chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / np.log10(highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop_1w = choppiness_index(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Get 1d data for KAMA calculation (faster than 1w for adaptive period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Kaufman Adaptive Moving Average (KAMA) on 1d close
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else np.array([0])
        volatility = pd.Series(close).rolling(window=period, min_periods=1).apply(lambda x: np.sum(np.abs(np.diff(x))) if len(x) > 1 else 0, raw=True).values
        
        er = np.zeros(len(close))
        for i in range(period, len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals = np.zeros(len(close))
        kama_vals[period-1] = close[period-1]
        for i in range(period, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_1d = kama(df_1d['close'].values)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # ATR for stop (14-period on 12h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of KAMA period (30), chop period (20), ATR (14)
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(chop_1w_aligned[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama_1d_aligned[i]
        chop_val = chop_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: price > KAMA and trending market (CHOP < 38.2)
            long_signal = (close_val > kama_val) and (chop_val < 38.2)
            # Short: price < KAMA and trending market (CHOP < 38.2)
            short_signal = (close_val < kama_val) and (chop_val < 38.2)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or regime change to ranging (CHOP > 61.8) or trend reversal (price < KAMA)
            if (low_val < long_stop) or (chop_val > 61.8) or (close_val < kama_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or regime change to ranging (CHOP > 61.8) or trend reversal (price > KAMA)
            if (high_val > short_stop) or (chop_val > 61.8) or (close_val > kama_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_With_Regime_Filter_v1"
timeframe = "12h"
leverage = 1.0