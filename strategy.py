#!/usr/bin/env python3
"""
4h_KAMA_Trend_DualTF_Confirmation
Hypothesis: Combines Kaufman Adaptive Moving Average (KAMA) on 4h for trend direction with 1d Williams %R for momentum and 1w EMA200 for long-term trend filter. Uses volume spike confirmation to filter false signals. Designed to capture trends in both bull and bear markets by requiring alignment across multiple timeframes. Targets 20-40 trades per year to minimize fee drag while maintaining edge.
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
    
    # Get 1-day and 1-week data for multi-timeframe confirmation
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 4h close (trend indicator)
    # Efficiency Ratio: |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of absolute changes
    # Handle the array shapes properly
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    
    # Calculate rolling sums for volatility
    volatility_sum = np.convolve(np.abs(np.diff(close, n=1)), np.ones(10), 'same')
    volatility_sum[:9] = np.nan  # Not enough data for first 9 periods
    
    er = np.where(volatility_sum != 0, change / volatility_sum, 0)
    # Fill NaNs from the beginning
    er = np.concatenate([np.full(9, np.nan), er[9:]])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # For EMA(2)
    slow_sc = 2 / (30 + 1)  # For EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start with close after enough data for ER
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate Williams %R on 1-day (momentum oscillator)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) != 0, williams_r, -50)
    
    # Calculate EMA200 on 1-week (long-term trend filter)
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # Already on 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume spike (>2.0x 20-period MA for strong confirmation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 200)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to KAMA (trend)
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # Williams %R conditions (oversold/overbought)
        wr_oversold = williams_r_aligned[i] < -80
        wr_overbought = williams_r_aligned[i] > -20
        
        # Long-term trend filter (1-week EMA200)
        long_term_uptrend = close[i] > ema_200_1w_aligned[i]
        long_term_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic:
        # Long: Price above KAMA (uptrend) + Williams %R oversold + long-term uptrend + volume spike
        long_entry = vol_confirm and price_above_kama and wr_oversold and long_term_uptrend
        
        # Short: Price below KAMA (downtrend) + Williams %R overbought + long-term downtrend + volume spike
        short_entry = vol_confirm and price_below_kama and wr_overbought and long_term_downtrend
        
        # Exit logic: Opposite conditions or loss of momentum
        long_exit = (price_below_kama or not wr_oversold or not long_term_uptrend)
        short_exit = (price_above_kama or not wr_overbought or not long_term_downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_DualTF_Confirmation"
timeframe = "4h"
leverage = 1.0