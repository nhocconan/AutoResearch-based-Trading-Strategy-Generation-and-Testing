#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d RSI filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion.
# Daily RSI filter ensures we only trade when momentum aligns with higher timeframe.
# Volume confirmation adds conviction to reversals.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in both bull and mean-reverting markets by fading extremes.
name = "12h_WilliamsR_1dRSI_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14-period) using previous period's data to avoid look-ahead
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().shift(1).values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate daily RSI (14-period)
    close_d = df_1d['close'].values
    delta = np.diff(close_d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    # Wilder's smoothing
    for i in range(1, len(gain)):
        if np.isnan(avg_gain[i-1]):
            avg_gain[i] = np.nanmean(gain[1:i+1]) if i >= 14 else np.nan
            avg_loss[i] = np.nanmean(loss[1:i+1]) if i >= 14 else np.nan
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND RSI not overbought (< 70) AND volume confirmation
            long_signal = williams_r[i] < -80 and rsi_aligned[i] < 70
            if vol_confirm and long_signal:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND RSI not oversold (> 30) AND volume confirmation
            elif vol_confirm and williams_r[i] > -20 and rsi_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) OR RSI overbought (>= 70)
            exit_condition = williams_r[i] > -50 or rsi_aligned[i] >= 70
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) OR RSI oversold (<= 30)
            exit_condition = williams_r[i] < -50 or rsi_aligned[i] <= 30
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals