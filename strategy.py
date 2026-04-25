#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Regime_v1
Hypothesis: Trade 12h Camarilla R1/S3 breakouts with 1d EMA50 trend filter and choppiness regime.
- Trend filter: price > 1d EMA50 = bullish, price < 1d EMA50 = bearish.
- In bullish 1d trend: buy breakouts above R1, sell breakdowns below S1.
- In bearish 1d trend: sell breakdowns below S1, buy breakouts above R1 (continuation logic).
- Choppiness regime filter: only trade when CHOP(14) > 61.8 (range-bound market) to avoid whipsaws in strong trends.
- Volume confirmation: require volume > 1.5x 20-period average to filter low-momentum breakouts.
- Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
- Works in both bull and bear: 1d trend filter captures major moves, chop filter avoids false signals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = df_1d['close'].values[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    atr_1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_1[0] = high[0] - low[0]  # first bar
    sum_atr_1 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    n_val = 14
    chop = 100 * np.log10(sum_atr_1 / (n_val * np.log(n_val))) / np.log10(n_val)
    
    # Volume spike confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(50) and chop/volume MA
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using EMA50
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Choppiness regime: only trade in ranging markets (CHOP > 61.8)
        in_range_regime = chop[i] > 61.8
        
        if position == 0:
            # Breakout logic: trade in direction of 1d trend, only in ranging regime
            long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_spike[i] and in_range_regime
            short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_spike[i] and in_range_regime
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal or mean reversion to pivot
            exit_signal = (not htf_1d_bullish) or (close[i] < pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or mean reversion to pivot
            exit_signal = htf_1d_bullish or (close[i] > pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0