#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly Donchian breakout + daily RSI filter + volume confirmation.
# Weekly Donchian (10-period) defines trend, daily RSI(14) < 40 for long / > 60 for short adds mean-reversion edge within trend.
# Volume > 1.5x 20-period average confirms breakout strength. Session filter 08-20 UTC to avoid low-liquidity hours.
# Designed for ~20-30 trades/year to avoid fee drag. Works in bull (trend + pullback) and bear (counter-trend bounces).

name = "6d_1w_Donchian10_1d_RSI14_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get weekly data for Donchian breakout (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Weekly Donchian channels: 10-period high/low
    high_10_1w = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    low_10_1w = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    high_10_1w_aligned = align_htf_to_ltf(prices, df_1w, high_10_1w)
    low_10_1w_aligned = align_htf_to_ltf(prices, df_1w, low_10_1w)
    
    # Get daily data for RSI (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # RSI(14) calculation
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(high_10_1w_aligned[i]) or np.isnan(low_10_1w_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly Donchian high AND RSI < 40 (oversold) with volume
            if (close[i] > high_10_1w_aligned[i] and 
                rsi_1d_aligned[i] < 40 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly Donchian low AND RSI > 60 (overbought) with volume
            elif (close[i] < low_10_1w_aligned[i] and 
                  rsi_1d_aligned[i] > 60 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low or RSI > 60
            if close[i] < low_10_1w_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high or RSI < 40
            if close[i] > high_10_1w_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals