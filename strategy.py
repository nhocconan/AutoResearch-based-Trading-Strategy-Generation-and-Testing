#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeRegimeFilter_v1
Hypothesis: On 6h timeframe, Elder Ray Bull/Bear Power combined with 1d EMA trend filter and volume regime (high/low volatility) provides robust signals in both bull and bear markets. Bull Power > 0 and Bear Power < 0 indicate buying/selling pressure. Uses 1d EMA50 for trend regime and ATR ratio for volatility filter to avoid choppy markets. Targets 80-120 trades over 4 years (20-30/year) with discrete sizing (0.0, ±0.25) to minimize fee churn. Works in bull via trend-following longs and in bear via mean-reversion shorts at extremes.
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
    
    # Get 1d data for HTF trend and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend regime
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) on 1d for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate ATR(14) on 6h for stops
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Elder Ray Bull Power and Bear Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate volume ratio (current / 50-period average) for volume regime
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(50), 1d ATR(14), EMA13, ATR(14), volume MA(50)
    start_idx = max(50, 14, 13, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        # Volatility regime: ATR ratio (current 6h ATR / 1d ATR) - high vol > 1.2, low vol < 0.8
        atr_ratio = atr_14[i] / np.maximum(atr_14_1d_aligned[i], 1e-10)
        high_vol = atr_ratio > 1.2
        low_vol = atr_ratio < 0.8
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND price above 1d EMA50 (bullish trend) AND high volume conviction
            long_signal = (bull_power[i] > 0) and (close_val > ema_50_1d_aligned[i]) and high_vol
            
            # Short: Bear Power < 0 (selling pressure) AND price below 1d EMA50 (bearish trend) AND high volume conviction
            short_signal = (bear_power[i] < 0) and (close_val < ema_50_1d_aligned[i]) and high_vol
            
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
            # Exit: Bull Power turns negative OR price below 1d EMA50 OR ATR stoploss (2.5x)
            if (bull_power[i] <= 0) or (close_val < ema_50_1d_aligned[i]) or (close_val < entry_price - 2.5 * atr_14[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power turns positive OR price above 1d EMA50 OR ATR stoploss (2.5x)
            if (bear_power[i] >= 0) or (close_val > ema_50_1d_aligned[i]) or (close_val > entry_price + 2.5 * atr_14[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeRegimeFilter_v1"
timeframe = "6h"
leverage = 1.0