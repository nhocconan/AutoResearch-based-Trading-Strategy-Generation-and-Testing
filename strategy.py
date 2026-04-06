#!/usr/bin/env python3
"""
1h RSI(14) extreme with 4h EMA(50) trend and 1d volume filter
Hypothesis: RSI extremes on 1h provide mean-reversion entries, filtered by 4h EMA50 for trend bias and 1d volume for confirmation. Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend). Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi14_extreme_4h_ema50_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)
    
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, n):
            avg_gain[i] = (gain[i] + avg_gain[i-1] * 13) / 14
            avg_loss[i] = (loss[i] + avg_loss[i-1] * 13) / 14
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100.0
    
    # Get 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # EMA50 on 4h close
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 48) / 50
    
    # 4h trend: above EMA50 = bullish, below = bearish
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    
    # Align 4h trend to 1h timeframe
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 1d volume MA to 1h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = 60  # Need enough data for RSI and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Volume filter: current 1h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 1h: approx 1/24 of 1d volume (since 24x 1h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 24.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss (using 2*ATR approximation via price change)
        if position == 1:  # long position
            # Exit: RSI returns to neutral OR against 4h trend
            if (rsi[i] >= 50 or
                trend_4h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.20
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: RSI returns to neutral OR against 4h trend
            if (rsi[i] <= 50 or
                trend_4h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.20
            bars_since_exit += 1
        else:
            # Look for entries with minimum bars between trades
            if bars_since_exit >= 6:  # Minimum 6 bars (6h) between trades
                # Mean reversion entries: RSI extremes with 4h trend + volume
                rsi_oversold = rsi[i] < 30
                rsi_overbought = rsi[i] > 70
                
                # Long: RSI oversold with bullish 4h trend + volume
                if rsi_oversold and trend_4h_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: RSI overbought with bearish 4h trend + volume
                elif rsi_overbought and trend_4h_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals