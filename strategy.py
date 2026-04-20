#!/usr/bin/env python3
"""
4h_1d_RSI_Divergence_Confluence
Hypothesis: Trade RSI divergences on 4h timeframe confirmed by 1d trend and volume spikes.
Look for bullish divergence (price makes lower low, RSI makes higher low) in uptrend (price > 1d EMA50) or bearish divergence (price makes higher high, RSI makes lower high) in downtrend (price < 1d EMA50).
Volume spike confirms institutional participation. Works in bull/bear: divergences signal reversals, volume validates strength.
Target: 50-100 total trades over 4 years (12-25/year) with position size 0.25.
"""

name = "4h_1d_RSI_Divergence_Confluence"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper handling of edge cases."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    # Wilder's smoothing
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    
    for i in range(period, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[:] = np.nan
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50_1d[i-1] * (49 / (50 + 1)))
    
    # Calculate 1d RSI for divergence detection
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Align daily indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 4h RSI for entry timing
    close_4h = prices['close'].values
    rsi_4h = calculate_rsi(close_4h, 14)
    
    # Calculate 4h volume average for spike detection (20-period)
    vol_4h = prices['volume'].values
    vol_avg_4h = np.full_like(vol_4h, np.nan)
    for i in range(len(vol_4h)):
        if i >= 19:  # 20-period average
            vol_avg_4h[i] = np.mean(vol_4h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get current values
        current_close = prices['close'].iloc[i]
        current_rsi_4h = rsi_4h[i]
        current_volume = prices['volume'].iloc[i]
        trend = ema_50_1d_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        
        # Skip if any essential data is missing
        if (np.isnan(trend) or np.isnan(rsi_1d_val) or 
            np.isnan(current_rsi_4h) or np.isnan(vol_avg_4h[i])):
            continue
        
        # Volume spike: current volume > 1.8x 4h average volume
        vol_spike = current_volume > 1.8 * vol_avg_4h[i]
        
        if position == 0:
            # Look for bullish divergence: price makes lower low, RSI makes higher low
            # Need to look back at least 5 periods for divergence
            if i >= 5:
                # Find recent low in price and RSI
                lookback = 5
                price_low_idx = i - np.argmin(close_4h[i-lookback:i+1])
                rsi_low_idx = i - np.argmin(rsi_4h[i-lookback:i+1])
                
                # Bullish divergence: price lower low, RSI higher low
                if (price_low_idx == i and rsi_low_idx < i and  # current bar is price low
                    close_4h[i] < close_4h[price_low_idx+1] and  # price made lower low
                    rsi_4h[i] > rsi_4h[rsi_low_idx+1] and       # RSI made higher low
                    current_close > trend and                   # price above 1d EMA50 (uptrend)
                    rsi_1d_val < 50 and                         # 1d RSI not overbought
                    vol_spike):                                 # volume confirmation
                    signals[i] = 0.25
                    position = 1
                # Bearish divergence: price makes higher high, RSI makes lower high
                elif (price_low_idx < i and rsi_low_idx == i and  # current bar is RSI high
                      close_4h[i] > close_4h[price_low_idx-1] and  # price made higher high
                      rsi_4h[i] < rsi_4h[rsi_low_idx-1] and       # RSI made lower high
                      current_close < trend and                   # price below 1d EMA50 (downtrend)
                      rsi_1d_val > 50 and                         # 1d RSI not oversold
                      vol_spike):                                 # volume confirmation
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend change or divergence failure
            if (current_rsi_4h > 70 or 
                current_close < trend * 0.98 or  # price falls below trend
                (current_rsi_4h < 40 and rsi_4h[i-1] >= 40)):  # RSI drops below 40
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or trend change or divergence failure
            if (current_rsi_4h < 30 or 
                current_close > trend * 1.02 or  # price rises above trend
                (current_rsi_4h > 60 and rsi_4h[i-1] <= 60)):  # RSI rises above 60
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals