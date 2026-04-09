#!/usr/bin/env python3
# 1d_camarilla_pivot_volume_regime_v1
# Hypothesis: Daily strategy using Camarilla pivot levels with volume confirmation and 1w HMA trend filter.
# Long: Price touches or breaks above Camarilla H3 level, volume > 1.5x 20-day average, price > 1w HMA(21).
# Short: Price touches or breaks below Camarilla L3 level, volume > 1.5x 20-day average, price < 1w HMA(21).
# Exit: Opposite Camarilla level touch or volume divergence.
# Uses 1w HMA for higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation filters weak breakouts. Target: 7-25 trades/year (30-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w HMA(21) for trend filter (MTF)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla pivot levels
    camarilla_high = np.zeros(n)
    camarilla_low = np.zeros(n)
    camarilla_h3 = np.zeros(n)
    camarilla_l3 = np.zeros(n)
    camarilla_h4 = np.zeros(n)
    camarilla_l4 = np.zeros(n)
    
    for i in range(1, n):
        # Previous day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla pivot calculations
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        camarilla_high[i] = pivot + (range_val * 1.1 / 2)
        camarilla_low[i] = pivot - (range_val * 1.1 / 2)
        camarilla_h3[i] = pivot + (range_val * 1.1 / 4)
        camarilla_l3[i] = pivot - (range_val * 1.1 / 4)
        camarilla_h4[i] = pivot + (range_val * 1.1 / 2)
        camarilla_l4[i] = pivot - (range_val * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price touches/below L3 OR volume divergence (price up but volume down)
            if close[i] <= camarilla_l3[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price touches/above H3 OR volume divergence (price down but volume down)
            if close[i] >= camarilla_h3[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price touches/above H3, volume confirmed, price > 1w HMA
            if (close[i] >= camarilla_h3[i] and volume_confirmed and close[i] > hma_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches/below L3, volume confirmed, price < 1w HMA
            elif (close[i] <= camarilla_l3[i] and volume_confirmed and close[i] < hma_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Weighted moving average function
    def wma(data, window):
        if len(data) < window:
            return np.full_like(data, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    # Calculate WMAs
    wma_full = wma(close, period)
    wma_half = wma(close, half_period)
    
    # Handle array alignment
    if len(wma_half) == 0 or len(wma_full) == 0:
        return np.full_like(close, np.nan)
    
    # 2*WMA(half) - WMA(full)
    # Need to align arrays properly
    diff = 2 * wma_half[-len(wma_full):] - wma_full
    
    # WMA of the difference with sqrt period
    hma_raw = wma(diff, sqrt_period)
    
    # Pad to original length
    hma = np.full_like(close, np.nan)
    start_idx = len(close) - len(hma_raw)
    if start_idx >= 0 and len(hma_raw) > 0:
        hma[start_idx:] = hma_raw
    
    return hma