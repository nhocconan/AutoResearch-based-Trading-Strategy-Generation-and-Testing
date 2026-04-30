#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h for trend direction (HMA21) and 1d for volume regime (above/below 20-period MA)
# Enter longs when: price > 4h HMA21 AND 1d volume > 20-period MA (bullish regime)
# Enter shorts when: price < 4h HMA21 AND 1d volume < 20-period MA (bearish regime)
# Uses session filter (08-20 UTC) to reduce noise. Discrete sizing 0.20 to minimize fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) by requiring both trend and volume regime alignment.

name = "1h_HMA21_Trend_1dVolumeRegime_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HMA21 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    hma_21 = calculate_hma(df_4h['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21)
    
    # 1d volume regime: above/below 20-period MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_regime = df_1d['volume'].values > vol_ma_20  # True = bullish regime, False = bearish regime
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # warmup for HMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(hma_21_aligned[i]) or np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_hma = hma_21_aligned[i]
        curr_vol_regime = volume_regime_aligned[i] > 0.5  # Convert back to boolean
        
        if position == 0:  # Flat - look for new entries
            # Bullish: price above 4h HMA AND bullish volume regime
            if curr_close > curr_hma and curr_vol_regime:
                signals[i] = 0.20
                position = 1
            # Bearish: price below 4h HMA AND bearish volume regime
            elif curr_close < curr_hma and not curr_vol_regime:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price crosses below 4h HMA (trend change)
            if curr_close < curr_hma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price crosses above 4h HMA (trend change)
            if curr_close > curr_hma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(values)
    for i in range(len(values)):
        if i < half_period - 1:
            wma_half[i] = np.nan
        else:
            start = i - half_period + 1
            weights = np.arange(1, half_period + 1)
            wma_half[i] = np.dot(values[start:i+1], weights) / weights.sum()
    
    # WMA of full period
    wma_full = np.zeros_like(values)
    for i in range(len(values)):
        if i < period - 1:
            wma_full[i] = np.nan
        else:
            start = i - period + 1
            weights = np.arange(1, period + 1)
            wma_full[i] = np.dot(values[start:i+1], weights) / weights.sum()
    
    # Raw HMA = 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA = WMA(sqrt_period) of raw_hma
    hma = np.zeros_like(values)
    for i in range(len(values)):
        if i < sqrt_period - 1:
            hma[i] = np.nan
        else:
            start = i - sqrt_period + 1
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.dot(raw_hma[start:i+1], weights) / weights.sum()
    
    return hma