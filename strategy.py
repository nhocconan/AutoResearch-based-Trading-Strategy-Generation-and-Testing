#!/usr/bin/env python3
"""
4h_HTF_CounterTrend_Momentum
4-hour counter-trend momentum strategy using 1-week RSI extremes and 1-day volume confirmation.
Long when: 1w RSI < 25 (deep oversold) + 1d volume > 1.5x 20-day average + price > 10-period EMA.
Short when: 1w RSI > 75 (overbought) + 1d volume > 1.5x 20-day average + price < 10-period EMA.
Exit when: 1w RSI crosses back to neutral (40-60 range).
Position size: 0.25. Target: 20-50 trades/year.
Uses 1w RSI for extreme condition, 1d volume for momentum confirmation, 4h EMA for trend alignment.
Works in both bull/bear: mean reversion in extremes, volume ensures momentum behind reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1w data for RSI
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w RSI(14)
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[:14] = np.nan  # Not enough data
    
    # Align 1w RSI to 4h
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        volume_ma20_1d[i] = np.mean(volume_1d[i-19:i+1])
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # Calculate 4h EMA10 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema10_4h = np.full_like(close_4h, np.nan)
    for i in range(9, len(close_4h)):
        ema10_4h[i] = (close_4h[i] * 2 + ema10_4h[i-1] * 9) / 11 if i > 9 else np.mean(close_4h[:i+1])
    ema10_4h_aligned = align_htf_to_ltf(prices, df_4h, ema10_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute current 4h close price (approximation using latest available)
    # For each 4h bar, we use the last 4h close available
    last_4h_close_idx = -1
    last_4h_close_val = np.nan
    
    for i in range(40, n):  # warmup for indicators
        # Update last known 4h close (simplified: assume we can track it)
        # In practice, we use the aligned EMA as trend proxy
        
        # Skip if any required data is not available
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or 
            np.isnan(ema10_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume aligned to 4h
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        volume_filter = vol_1d_current > (1.5 * volume_ma20_1d_aligned[i])
        
        # RSI extremes
        rsi_oversold = rsi_1w_aligned[i] < 25
        rsi_overbought = rsi_1w_aligned[i] > 75
        rsi_neutral = (rsi_1w_aligned[i] >= 40) & (rsi_1w_aligned[i] <= 60)
        
        # Price vs EMA10 trend filter
        # Use close price vs EMA10 (we approximate current close with EMA for simplicity)
        price_above_ema = close[i] > ema10_4h_aligned[i]  # Simplified
        price_below_ema = close[i] < ema10_4h_aligned[i]  # Simplified
        
        if position == 0:
            # Long: 1w RSI oversold + volume surge + price above EMA10
            if rsi_oversold and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: 1w RSI overbought + volume surge + price below EMA10
            elif rsi_overbought and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1w RSI returns to neutral
            if rsi_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1w RSI returns to neutral
            if rsi_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_CounterTrend_Momentum"
timeframe = "4h"
leverage = 1.0