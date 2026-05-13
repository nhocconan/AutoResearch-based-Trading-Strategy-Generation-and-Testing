#!/usr/bin/env python3
# 6h_WMA_Cross_1dATR_Volume_Regime
# Hypothesis: Weighted Moving Average crossovers on 6h chart filtered by 1-day ATR volatility regime and volume confirmation.
# Uses WMA(9)/WMA(21) cross for trend changes, with entry only when ATR(14) is in normal range (not too high/low volatility).
# Works in bull markets (trend following on golden crosses) and bear markets (trend following on death crosses).
# Target: 15-30 trades/year to minimize fee drag on 6h timeframe.

name = "6h_WMA_Cross_1dATR_Volume_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wma(values, window):
    """Calculate Weighted Moving Average"""
    if len(values) < window:
        return np.full_like(values, np.nan)
    weights = np.arange(1, window + 1)
    wma_values = np.full_like(values, np.nan)
    for i in range(window - 1, len(values)):
        wma_values[i] = np.dot(values[i - window + 1:i + 1], weights) / weights.sum()
    return wma_values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate WMA(9) and WMA(21) for crossover signals
    wma_9 = wma(close, 9)
    wma_21 = wma(close, 21)
    
    # Calculate 1-day ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan  # First value has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_14[i] = np.nanmean(tr[0:14])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate ATR percentile over 50 periods to define normal volatility regime
    atr_percentile = np.full_like(atr_14_aligned, np.nan)
    for i in range(49, len(atr_14_aligned)):
        if not np.isnan(atr_14_aligned[i]):
            window_data = atr_14_aligned[i-49:i+1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                percentile_rank = (np.sum(valid_data <= atr_14_aligned[i]) / len(valid_data)) * 100
                atr_percentile[i] = percentile_rank
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_confirmed = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if ATR percentile not calculated (volatility regime unclear)
        if np.isnan(atr_percentile[i]):
            signals[i] = 0.0
            continue
            
        # Define normal volatility regime: ATR between 20th and 80th percentile
        normal_volatility = (atr_percentile[i] >= 20) and (atr_percentile[i] <= 80)
        
        if position == 0:
            # LONG: WMA(9) crosses above WMA(21) with normal volatility and volume confirmation
            if (not np.isnan(wma_9[i-1]) and not np.isnan(wma_21[i-1]) and
                not np.isnan(wma_9[i]) and not np.isnan(wma_21[i]) and
                wma_9[i-1] <= wma_21[i-1] and wma_9[i] > wma_21[i] and
                normal_volatility and volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: WMA(9) crosses below WMA(21) with normal volatility and volume confirmation
            elif (not np.isnan(wma_9[i-1]) and not np.isnan(wma_21[i-1]) and
                  not np.isnan(wma_9[i]) and not np.isnan(wma_21[i]) and
                  wma_9[i-1] >= wma_21[i-1] and wma_9[i] < wma_21[i] and
                  normal_volatility and volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: WMA(9) crosses below WMA(21) or volatility becomes extreme
            if (not np.isnan(wma_9[i-1]) and not np.isnan(wma_21[i-1]) and
                not np.isnan(wma_9[i]) and not np.isnan(wma_21[i]) and
                wma_9[i-1] >= wma_21[i-1] and wma_9[i] < wma_21[i]) or \
               (atr_percentile[i] < 20) or (atr_percentile[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: WMA(9) crosses above WMA(21) or volatility becomes extreme
            if (not np.isnan(wma_9[i-1]) and not np.isnan(wma_21[i-1]) and
                not np.isnan(wma_9[i]) and not np.isnan(wma_21[i]) and
                wma_9[i-1] <= wma_21[i-1] and wma_9[i] > wma_21[i]) or \
               (atr_percentile[i] < 20) or (atr_percentile[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals