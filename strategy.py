# 4h_Stochastic_Squeeze_Reversal_v1
# Mean reversion strategy using Bollinger Band squeeze + Stochastic oscillator
# Exploits volatility contraction followed by mean reversion in ranging markets
# Works in both bull and bear markets by focusing on range-bound periods
# Uses 1d Bollinger Bands for regime detection and 4h Stochastic for entry timing
# Target: 20-50 trades/year to minimize fee drag

#!/usr/bin/env python3
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
    
    # === 1d Bollinger Bands for regime detection ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate BB(20,2) on daily timeframe
    bb_length = 20
    bb_std = 2.0
    bb_sum = np.zeros(len(close_1d))
    bb_mean = np.zeros(len(close_1d))
    bb_std_dev = np.zeros(len(close_1d))
    bb_upper = np.zeros(len(close_1d))
    bb_lower = np.zeros(len(close_1d))
    bb_width = np.zeros(len(close_1d))
    
    # Initialize
    if len(close_1d) > 0:
        bb_sum[0] = close_1d[0]
        bb_mean[0] = close_1d[0]
        bb_std_dev[0] = 0.0
        bb_upper[0] = close_1d[0]
        bb_lower[0] = close_1d[0]
        bb_width[0] = 0.0
    
    for i in range(1, len(close_1d)):
        if i < bb_length:
            bb_sum[i] = bb_sum[i-1] + close_1d[i]
            bb_mean[i] = bb_sum[i] / (i + 1)
            # Calculate std dev for available data
            if i > 0:
                sq_diff = np.sum((close_1d[0:i+1] - bb_mean[i]) ** 2)
                bb_std_dev[i] = np.sqrt(sq_diff / (i + 1))
            else:
                bb_std_dev[i] = 0.0
        else:
            bb_sum[i] = bb_sum[i-1] + close_1d[i] - close_1d[i-bb_length]
            bb_mean[i] = bb_sum[i] / bb_length
            # Standard deviation for the window
            window_data = close_1d[i-bb_length+1:i+1]
            bb_std_dev[i] = np.std(window_data)
        
        bb_upper[i] = bb_mean[i] + bb_std_dev[i] * bb_std
        bb_lower[i] = bb_mean[i] - bb_std_dev[i] * bb_std
        bb_width[i] = bb_upper[i] - bb_lower[i]
    
    # === 4h Stochastic Oscillator for entry timing ===
    # Calculate Stochastic(14,3,3) on 4h data
    stoch_k_period = 14
    stoch_d_period = 3
    stoch_slowing = 3
    
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - stoch_k_period + 1)
        lowest_low[i] = np.min(low[start_idx:i+1])
        highest_high[i] = np.max(high[start_idx:i+1])
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    stoch_raw = np.where(denominator != 0, 
                         (close - lowest_low) / denominator * 100, 
                         50.0)
    
    # %K with slowing
    stoch_k = np.zeros(n)
    for i in range(n):
        if i < stoch_slowing - 1:
            stoch_k[i] = np.mean(stoch_raw[0:i+1]) if i >= 0 else 50.0
        else:
            stoch_k[i] = np.mean(stoch_raw[i-stoch_slowing+1:i+1])
    
    # %D (SMA of %K)
    stoch_d = np.zeros(n)
    for i in range(n):
        if i < stoch_d_period - 1:
            stoch_d[i] = np.mean(stoch_k[0:i+1]) if i >= 0 else 50.0
        else:
            stoch_d[i] = np.mean(stoch_k[i-stoch_d_period+1:i+1])
    
    # === Align HTF data ===
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate Bollinger Band width percentile for squeeze detection
    bb_width_percentile = np.zeros(n)
    lookback_period = 50  # Look back 50 periods for percentile calculation
    
    for i in range(n):
        start_idx = max(0, i - lookback_period + 1)
        if i >= lookback_period - 1:
            historical_widths = bb_width_aligned[start_idx:i+1]
            current_width = bb_width_aligned[i]
            if len(historical_widths) > 0:
                percentile = np.sum(historical_widths <= current_width) / len(historical_widths) * 100
                bb_width_percentile[i] = percentile
            else:
                bb_width_percentile[i] = 50.0
        else:
            bb_width_percentile[i] = 50.0
    
    # === Signal Generation ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    warmup = max(50, stoch_k_period + stoch_d_period + stoch_slowing)
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Bollinger Band squeeze condition: low volatility regime
        # Squeeze when BB width is in lower 20% percentile (volatile contraction)
        is_squeeze = bb_width_percentile[i] < 20
        
        # Stochastic oversold/overbought conditions
        is_oversold = stoch_k[i] < 20 and stoch_d[i] < 20
        is_overbought = stoch_k[i] > 80 and stoch_d[i] > 80
        
        # Price near Bollinger Bands for mean reversion setup
        bb_middle = (bb_upper_aligned[i] + bb_lower_aligned[i]) / 2
        bb_range = bb_upper_aligned[i] - bb_lower_aligned[i]
        if bb_range > 0:
            price_position = (close[i] - bb_lower_aligned[i]) / bb_range
        else:
            price_position = 0.5
        
        near_lower_band = price_position < 0.2  # Near lower Bollinger Band
        near_upper_band = price_position > 0.8   # Near upper Bollinger Band
        
        # Entry logic: mean reversion from Bollinger Bands during squeeze
        if position == 0 and is_squeeze:
            # Long setup: price near lower band during squeeze + stochastic oversold
            if near_lower_band and is_oversold:
                signals[i] = 0.25
                position = 1
                continue
            # Short setup: price near upper band during squeeze + stochastic overbought
            elif near_upper_band and is_overbought:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: mean reversion complete or volatility expansion
        elif position == 1:
            # Exit long: price reaches middle band or stochastic overbought
            if close[i] >= bb_middle or stoch_k[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches middle band or stochastic oversold
            if close[i] <= bb_middle or stoch_k[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Stochastic_Squeeze_Reversal_v1"
timeframe = "4h"
leverage = 1.0