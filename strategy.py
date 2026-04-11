#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-week and 1-day momentum divergence + volume confirmation.
# Uses weekly RSI divergence to identify trend exhaustion and daily RSI for entry timing.
# Long when: weekly RSI shows bullish divergence (price makes lower low, RSI makes higher low) + daily RSI < 30 + volume > 1.5x average.
# Short when: weekly RSI shows bearish divergence (price makes higher high, RSI makes lower high) + daily RSI > 70 + volume > 1.5x average.
# Weekly divergence filter reduces whipsaw in sideways markets and improves win rate.
# Designed for 20-40 trades/year with focus on reversal points in both bull and bear markets.

name = "6h_1w1d_rsi_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 14 or len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = np.full_like(gain_1d, np.nan, dtype=float)
    avg_loss_1d = np.full_like(loss_1d, np.nan, dtype=float)
    
    for i in range(14, len(gain_1d)):
        if i == 14:
            avg_gain_1d[i] = np.mean(gain_1d[1:15])
            avg_loss_1d[i] = np.mean(loss_1d[1:15])
        else:
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    
    rs_1d = np.where(avg_loss_1d != 0, avg_gain_1d / avg_loss_1d, 0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = np.full_like(gain_1w, np.nan, dtype=float)
    avg_loss_1w = np.full_like(loss_1w, np.nan, dtype=float)
    
    for i in range(14, len(gain_1w)):
        if i == 14:
            avg_gain_1w[i] = np.mean(gain_1w[1:15])
            avg_loss_1w[i] = np.mean(loss_1w[1:15])
        else:
            avg_gain_1w[i] = (avg_gain_1w[i-1] * 13 + gain_1w[i]) / 14
            avg_loss_1w[i] = (avg_loss_1w[i-1] * 13 + loss_1w[i]) / 14
    
    rs_1w = np.where(avg_loss_1w != 0, avg_gain_1w / avg_loss_1w, 0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Calculate weekly RSI divergence (lookback 3 periods)
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    
    # Find local minima and maxima in price (3-bar window)
    price_min_1w = np.full_like(close_1w, np.nan, dtype=float)
    price_max_1w = np.full_like(close_1w, np.nan, dtype=float)
    rsi_min_1w = np.full_like(rsi_1w, np.nan, dtype=float)
    rsi_max_1w = np.full_like(rsi_1w, np.nan, dtype=float)
    
    for i in range(2, len(close_1w)-2):
        # Local minimum: lower than neighbors
        if close_1w[i] <= close_1w[i-1] and close_1w[i] <= close_1w[i-2] and \
           close_1w[i] <= close_1w[i+1] and close_1w[i] <= close_1w[i+2]:
            price_min_1w[i] = close_1w[i]
            rsi_min_1w[i] = rsi_1w[i]
        # Local maximum: higher than neighbors
        if close_1w[i] >= close_1w[i-1] and close_1w[i] >= close_1w[i-2] and \
           close_1w[i] >= close_1w[i+1] and close_1w[i] >= close_1w[i+2]:
            price_max_1w[i] = close_1w[i]
            rsi_max_1w[i] = rsi_1w[i]
    
    # Find previous local min/max for divergence detection
    prev_price_min = np.roll(price_min_1w, 1)
    prev_rsi_min = np.roll(rsi_min_1w, 1)
    prev_price_max = np.roll(price_max_1w, 1)
    prev_rsi_max = np.roll(rsi_max_1w, 1)
    
    # Mark first element as NaN
    prev_price_min[0] = np.nan
    prev_rsi_min[0] = np.nan
    prev_price_max[0] = np.nan
    prev_rsi_max[0] = np.nan
    
    # Bullish divergence: current price makes lower low than previous low, but RSI makes higher low
    bull_div = ((price_min_1w < prev_price_min) & (rsi_min_1w > prev_rsi_min)) & \
               (~np.isnan(price_min_1w)) & (~np.isnan(prev_price_min)) & \
               (~np.isnan(rsi_min_1w)) & (~np.isnan(prev_rsi_min))
    
    # Bearish divergence: current price makes higher high than previous high, but RSI makes lower high
    bear_div = ((price_max_1w > prev_price_max) & (rsi_max_1w < prev_rsi_max)) & \
               (~np.isnan(price_max_1w)) & (~np.isnan(prev_price_max)) & \
               (~np.isnan(rsi_max_1w)) & (~np.isnan(prev_rsi_max))
    
    # Daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all indicators to 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bull_div_aligned = align_htf_to_ltf(prices, df_1w, bull_div)
    bear_div_aligned = align_htf_to_ltf(prices, df_1w, bear_div)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(bull_div_aligned[i]) or np.isnan(bear_div_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Long signal: bullish divergence + oversold RSI + volume
        long_signal = bull_div_aligned[i] and (rsi_1d_aligned[i] < 30) and vol_filter
        
        # Short signal: bearish divergence + overbought RSI + volume
        short_signal = bear_div_aligned[i] and (rsi_1d_aligned[i] > 70) and vol_filter
        
        # Exit signals
        exit_long = position == 1 and (rsi_1d_aligned[i] > 70 or not vol_filter)
        exit_short = position == -1 and (rsi_1d_aligned[i] < 30 or not vol_filter)
        
        # Update position and signals
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals