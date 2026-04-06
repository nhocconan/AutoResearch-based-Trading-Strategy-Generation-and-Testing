#!/usr/bin/env python3
"""
1d KAMA trend with 1w trend filter and volume confirmation
Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both bull and bear markets.
Filtered by 1w EMA trend for bias and volume confirmation for conviction. Target: 75-250 total trades over 4 years (19-62/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast = 2
    slow = 30
    lookback = 10
    
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(lookback, n):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Calculate SSC (Smoothing Constant)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[lookback] = close[lookback]
    for i in range(lookback+1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA50 on 1w close
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 48) / 50
    
    # 1w trend: above EMA50 = bullish, below = bearish
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    
    # Align 1w trend to 1d timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get 1w data for volume confirmation
    volume_1w = df_1w['volume'].values
    
    # 20-period average volume on 1w
    vol_ma_1w = np.full(len(volume_1w), np.nan)
    for i in range(20, len(volume_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-20:i])
    
    # Align volume MA to 1d timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 60  # Need enough data for KAMA and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 1d volume > 1.5x 1w average volume (scaled)
        # Scale 1w volume to 1d: approx 1/5 of 1w volume (since 5x 1d in 1w)
        vol_threshold = vol_ma_1w_aligned[i] / 5.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below KAMA OR against 1w trend
            # Stoploss: price drops 2*ATR below entry (using price volatility as proxy)
            price_change = np.abs(close[i] - close[i-1]) if i > 0 else 0
            avg_volatility = np.mean([np.abs(close[j] - close[j-1]) for j in range(max(1, i-9), i+1)]) if i >= 1 else 0
            atr_estimate = avg_volatility if avg_volatility > 0 else price_change
            
            if (close[i] < kama[i] or
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr_estimate):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above KAMA OR against 1w trend
            # Stoploss: price rises 2*ATR above entry
            price_change = np.abs(close[i] - close[i-1]) if i > 0 else 0
            avg_volatility = np.mean([np.abs(close[j] - close[j-1]) for j in range(max(1, i-9), i+1)]) if i >= 1 else 0
            atr_estimate = avg_volatility if avg_volatility > 0 else price_change
            
            if (close[i] > kama[i] or
                trend_1w_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr_estimate):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 5 bars flat
            if bars_since_entry >= 5:
                # Entry signals: price crosses KAMA with 1w trend + volume
                bull_cross = close[i] > kama[i] and close[i-1] <= kama[i-1]
                bear_cross = close[i] < kama[i] and close[i-1] >= kama[i-1]
                
                # Long: cross above KAMA with bullish 1w trend + volume
                if bull_cross and trend_1w_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: cross below KAMA with bearish 1w trend + volume
                elif bear_cross and trend_1w_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals