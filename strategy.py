#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day RSI filter and 12-hour Donchian breakout (20-period) with volume confirmation.
# Enters only during 08-20 UTC session. Uses strict conditions to limit trades (~15-30/year) and avoid overtrading.
# Trend-following in bull markets, avoids false signals in bear/chop via RSI filter and volume spike requirement.
name = "12h_1d_RSI14_Donchian20_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for RSI14 filter (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # RSI(14) calculation
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Get 12h data for Donchian20 breakout (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    # Donchian channels: 20-period high/low
    high_20_12h = np.full_like(high_12h, np.nan)
    low_20_12h = np.full_like(low_12h, np.nan)
    for i in range(20, len(high_12h)):
        high_20_12h[i] = np.max(high_12h[i-20:i])
        low_20_12h[i] = np.min(low_12h[i-20:i])
    high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        volume_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(high_20_12h_aligned[i]) or 
            np.isnan(low_20_12h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d RSI50 (bullish bias) AND breaks 12h Donchian high with volume
            if (rsi_14_1d_aligned[i] > 50 and 
                close[i] > high_20_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d RSI50 (bearish bias) AND breaks 12h Donchian low with volume
            elif (rsi_14_1d_aligned[i] < 50 and 
                  close[i] < low_20_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d RSI50 or 12h Donchian low
            if rsi_14_1d_aligned[i] < 50 or close[i] < low_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d RSI50 or 12h Donchian high
            if rsi_14_1d_aligned[i] > 50 or close[i] > high_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals