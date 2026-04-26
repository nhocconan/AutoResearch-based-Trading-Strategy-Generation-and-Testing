#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Regime_Filter_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) as primary trend filter on 1d timeframe.
Enter long when price crosses above KAMA with volume confirmation and choppy regime filter (Choppiness Index > 61.8).
Enter short when price crosses below KAMA with volume confirmation and choppy regime filter.
Uses ATR-based trailing stop (2.5x ATR) for risk management.
Designed for low trade frequency (~10-25/year) by requiring confluence: KAMA cross + volume spike + chop regime.
Works in ranging markets (chop filter enables mean reversion at extremes) and can capture trends when chop low.
Focus on BTC/ETH as primary targets.
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
    
    # Get 1d data for KAMA trend and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on 1d close
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # 10-period sum of abs changes
    volatility = np.concatenate([[np.nan]*9, volatility])  # align lengths
    
    # Avoid division by zero
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(TR,14) / (ATR(14) * 14)) / log10(14)
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    
    # Align 1d indicators to higher timeframe (prices timeframe)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)  # chop is concurrent
    
    # Volume confirmation: 2.0x median volume (50-period)
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    # ATR for stop (22-period on prices timeframe)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_22 = pd.Series(tr).rolling(window=22, min_periods=22).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of KAMA seed (10), chop (14), volume median (50), ATR (22)
    start_idx = max(10, 14, 50, 22)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_22[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_22_val = atr_22[i]
        
        if position == 0:
            # Long: price > KAMA, volume spike, choppy regime (CHOP > 61.8 = ranging)
            long_signal = (close_val > kama_val) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (chop_val > 61.8)
            # Short: price < KAMA, volume spike, choppy regime (CHOP > 61.8 = ranging)
            short_signal = (close_val < kama_val) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (chop_val > 61.8)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_22_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_22_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_22_val)
            # Exit: trailing stop hit or trend reversal (price < KAMA)
            if (low_val < long_stop) or (close_val < kama_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_22_val)
            # Exit: trailing stop hit or trend reversal (price > KAMA)
            if (high_val > short_stop) or (close_val > kama_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_Regime_Filter_v1"
timeframe = "1d"
leverage = 1.0