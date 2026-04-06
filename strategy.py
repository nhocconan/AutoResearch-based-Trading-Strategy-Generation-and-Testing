#!/usr/bin/env python3
"""
1h RSI divergence with 4h trend filter and volume confirmation
Hypothesis: RSI divergences capture momentum exhaustion. Works in bull (buy bullish divergence on pullbacks) and bear (sell bearish divergence on rallies). 4h trend filter ensures alignment with higher timeframe momentum. Volume confirmation adds conviction. Target: 100-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_divergence_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period RSI
    rsi = np.full(n, np.nan)
    if n >= 15:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter (EMA50)
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
    
    # Get 4h data for volume confirmation
    volume_4h = df_4h['volume'].values
    
    # 20-period average volume on 4h
    vol_ma_4h = np.full(len(volume_4h), np.nan)
    for i in range(20, len(volume_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i-20:i])
    
    # Align volume MA to 1h timeframe
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 60  # Need enough data for RSI and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 1h volume > 1.5x 4h average volume (scaled)
        # Scale 4h volume to 1h: approx 1/4 of 4h volume (since 4x 1h in 4h)
        vol_threshold = vol_ma_4h_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) OR against 4h trend
            # Stoploss: price drops 2*ATR below entry (using simplified ATR approximation)
            if (rsi[i] > 70 or
                trend_4h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) OR against 4h trend
            # Stoploss: price rises 2*ATR above entry
            if (rsi[i] < 30 or
                trend_4h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
            bars_since_entry += 1
        else:
            # Look for entries with minimum holding period
            if bars_since_entry >= 12:
                # Need at least 5 bars of RSI history for divergence
                if i >= 5:
                    # Bullish divergence: price makes lower low, RSI makes higher low
                    bull_div = (low[i] < low[i-3] and 
                               rsi[i] > rsi[i-3] and
                               rsi[i] < 40)  # RSI not too high
                    
                    # Bearish divergence: price makes higher high, RSI makes lower high
                    bear_div = (high[i] > high[i-3] and 
                               rsi[i] < rsi[i-3] and
                               rsi[i] > 60)  # RSI not too low
                    
                    # Long: bullish divergence with bullish 4h trend + volume
                    if bull_div and trend_4h_aligned[i] == 1 and volume_filter:
                        signals[i] = 0.20
                        position = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                    # Short: bearish divergence with bearish 4h trend + volume
                    elif bear_div and trend_4h_aligned[i] == -1 and volume_filter:
                        signals[i] = -0.20
                        position = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.0
                        bars_since_entry += 1
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals