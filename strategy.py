#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI divergence with daily volume confirmation and weekly trend filter.
# Uses RSI(14) divergences for mean reversion entries, volume spike to confirm institutional interest,
# and weekly EMA(50) trend filter to avoid counter-trend trades in strong trends.
# Designed for 20-50 trades/year with focus on BTC/ETH robustness.

name = "4h_1d1w_rsi_divergence_v1"
timeframe = "4h"
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
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate weekly EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 49) / 51
    
    # Align indicators to 4h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)  # RSI from daily but aligned to 4h
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Initialize signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Lookback period for divergence detection
    lookback = 10
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 2.0 * daily average volume
        vol_filter = volume[i] > 2.0 * vol_avg_aligned[i]
        
        # Determine weekly trend: above EMA50 = bullish, below = bearish
        is_bullish_trend = close[i] > ema_50_aligned[i]
        is_bearish_trend = close[i] < ema_50_aligned[i]
        
        # Bullish RSI divergence: price makes lower low, RSI makes higher low
        bull_div = False
        if i >= lookback:
            for j in range(1, lookback+1):
                if low[i-j] < low[i] and rsi_aligned[i-j] > rsi_aligned[i]:
                    bull_div = True
                    break
        
        # Bearish RSI divergence: price makes higher high, RSI makes lower high
        bear_div = False
        if i >= lookback:
            for j in range(1, lookback+1):
                if high[i-j] > high[i] and rsi_aligned[i-j] < rsi_aligned[i]:
                    bear_div = True
                    break
        
        # Entry conditions
        bull_entry = bull_div and vol_filter and is_bullish_trend
        bear_entry = bear_div and vol_filter and is_bearish_trend
        
        # Exit conditions: opposite divergence or RSI extreme
        bull_exit = (position == 1 and 
                    (rsi_aligned[i] > 70 or bear_div))
        bear_exit = (position == -1 and 
                    (rsi_aligned[i] < 30 or bull_div))
        
        # Update position and signals
        if bull_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and bull_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bear_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals