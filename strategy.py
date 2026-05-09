#!/usr/bin/env python3
# 4h_KAMA_Trend_Strength_With_Adaptive_Bandwidth
# Hypothesis: KAMA adapts to market noise—low volatility follows trend, high volatility mean-reverts.
# Uses KAMA direction (trend) + Bollinger Band width (volatility regime) + volume confirmation.
# In low volatility (BB width < 50th percentile), follow KAMA trend. In high volatility, mean-revert at Bollinger Bands.
# Works in bull/bear: adapts to regime. Volatility filter reduces whipsaws. Volume confirms institutional participation.

name = "4h_KAMA_Trend_Strength_With_Adaptive_Bandwidth"
timeframe = "4h"
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
    
    # Calculate KAMA (adaptive moving average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation over 10 periods
    er = np.full_like(close, np.nan)
    for i in range(10, n):
        directional_change = np.abs(close[i] - close[i-10])
        total_change = np.sum(np.abs(np.diff(close[i-9:i+1])))
        if total_change > 0:
            er[i] = directional_change / total_change
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    if n > 0:
        kama[0] = close[0]
        for i in range(1, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # Bollinger Bands (20, 2) for volatility regime
    bb_period = 20
    bb_std = 2
    sma = np.full_like(close, np.nan)
    bb_std_dev = np.full_like(close, np.nan)
    
    if n >= bb_period:
        for i in range(bb_period-1, n):
            sma[i] = np.mean(close[i-bb_period+1:i+1])
            bb_std_dev[i] = np.std(close[i-bb_period+1:i+1])
    
    bb_upper = sma + bb_std * bb_std_dev
    bb_lower = sma - bb_std * bb_std_dev
    bb_width = (bb_upper - bb_lower) / sma  # normalized width
    
    # Percentile of BB width for regime detection (use 50th percentile as threshold)
    bb_width_median = np.full_like(bb_width, np.nan)
    lookback = 50
    for i in range(lookback, n):
        if i >= lookback:
            window = bb_width[i-lookback+1:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                bb_width_median[i] = np.median(valid)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 10, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(sma[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_median[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime: low volatility (trend following) or high volatility (mean reversion)
        is_low_vol = bb_width[i] < bb_width_median[i]
        
        if position == 0:
            if is_low_vol:
                # Low volatility: follow KAMA trend
                if close[i] > kama[i] and volume_ratio[i] > 1.5:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and volume_ratio[i] > 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                # High volatility: mean reversion at Bollinger Bands
                if close[i] <= bb_lower[i] and volume_ratio[i] > 1.5:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= bb_upper[i] and volume_ratio[i] > 1.5:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            if is_low_vol:
                # Exit long trend: price below KAMA
                if close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit long mean reversion: price above SMA (mean)
                if close[i] >= sma[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            if is_low_vol:
                # Exit short trend: price above KAMA
                if close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit short mean reversion: price below SMA (mean)
                if close[i] <= sma[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals