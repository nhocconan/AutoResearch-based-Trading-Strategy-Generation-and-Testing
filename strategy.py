#!/usr/bin/env python3
"""
Hypothesis: 6-hour 1-week RSI divergence with 1-day trend filter and volume confirmation.
Long when price makes new low but RSI makes higher low (bullish divergence) with 1-day EMA50 rising and volume spike.
Short when price makes new high but RSI makes lower high (bearish divergence) with 1-day EMA50 falling and volume spike.
Exit when RSI crosses above 50 (long) or below 50 (short).
Weekly RSI divergence captures exhaustion points in trends, effective in both bull and bear markets.
1-day EMA50 filters for trend alignment, volume spike confirms institutional interest.
Designed for low trade frequency by requiring multiple confirmations and weekly signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with given period."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for RSI calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    rsi_1w = calculate_rsi(close_1w, 14)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for calculations
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bullish divergence: price makes new low but RSI makes higher low
            # Check for new 10-period low in price
            if i >= 10:
                price_low_10 = np.min(low[i-9:i+1])
                is_new_price_low = low[i] == price_low_10
                
                # Find RSI value at the same point as the price low
                # Look back up to 10 periods for where the price low occurred
                rsi_at_low = np.nan
                for j in range(10):
                    idx = i - j
                    if low[idx] == price_low_10 and not np.isnan(rsi_1w_aligned[idx]):
                        rsi_at_low = rsi_1w_aligned[idx]
                        break
                
                # Find previous RSI low (look back 11-20 periods)
                prev_rsi_low = np.nan
                if i >= 20:
                    price_low_20 = np.min(low[i-19:i-9])
                    for j in range(10, 20):
                        idx = i - j
                        if low[idx] == price_low_20 and not np.isnan(rsi_1w_aligned[idx]):
                            prev_rsi_low = rsi_1w_aligned[idx]
                            break
                
                bullish_div = (is_new_price_low and 
                              not np.isnan(rsi_at_low) and not np.isnan(prev_rsi_low) and
                              rsi_at_low > prev_rsi_low)
                
                if (bullish_div and 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                    signals[i] = 0.25
                    position = 1
            
            # Bearish divergence: price makes new high but RSI makes lower high
            # Check for new 10-period high in price
            if i >= 10:
                price_high_10 = np.max(high[i-9:i+1])
                is_new_price_high = high[i] == price_high_10
                
                # Find RSI value at the same point as the price high
                rsi_at_high = np.nan
                for j in range(10):
                    idx = i - j
                    if high[idx] == price_high_10 and not np.isnan(rsi_1w_aligned[idx]):
                        rsi_at_high = rsi_1w_aligned[idx]
                        break
                
                # Find previous RSI high (look back 11-20 periods)
                prev_rsi_high = np.nan
                if i >= 20:
                    price_high_20 = np.max(high[i-19:i-9])
                    for j in range(10, 20):
                        idx = i - j
                        if high[idx] == price_high_20 and not np.isnan(rsi_1w_aligned[idx]):
                            prev_rsi_high = rsi_1w_aligned[idx]
                            break
                
                bearish_div = (is_new_price_high and 
                              not np.isnan(rsi_at_high) and not np.isnan(prev_rsi_high) and
                              rsi_at_high < prev_rsi_high)
                
                if (bearish_div and 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: RSI crosses 50 level
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses above 50
                if rsi_1w_aligned[i] > 50 and rsi_1w_aligned[i-1] <= 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses below 50
                if rsi_1w_aligned[i] < 50 and rsi_1w_aligned[i-1] >= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_1W_RSI_Divergence_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0