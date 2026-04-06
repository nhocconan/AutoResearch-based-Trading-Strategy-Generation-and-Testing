#!/usr/bin/env python3
"""
6h Williams %R reversal with 1d volume confirmation and trend filter
Hypothesis: Williams %R captures short-term overbought/oversold conditions. In 6h timeframe,
extreme readings (< -80 or > -20) combined with 1d volume surge and 1d trend filter
provide high-probability reversals. Works in bull (buy oversold in uptrend) and bear
(sell overbought in downtrend). Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williamsr_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Williams %R (14-period) on 6h data
    willr = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            willr[i] = -50  # neutral when no range
    
    # Get 1d data for trend filter (close vs SMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # SMA50 on daily close
    sma_1d = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        sma_1d[i] = np.mean(close_1d[i-49:i+1])
    
    # Daily trend: above SMA50 = bullish, below = bearish
    daily_trend = np.where(close_1d > sma_1d, 1, -1)
    
    # Align daily trend to 6h timeframe
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on daily
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for Williams %R and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(willr[i]) or 
            np.isnan(daily_trend_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x daily average volume (scaled)
        # Scale daily volume to 6h: approx 1/4 of daily volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R returns from oversold OR against daily trend
            # Stoploss: price drops 2*ATR below entry
            if (willr[i] > -50 or  # exited oversold
                daily_trend_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: Williams %R returns from overbought OR against daily trend
            # Stoploss: price rises 2*ATR above entry
            if (willr[i] < -50 or  # exited overbought
                daily_trend_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 8 bars flat
            if bars_since_entry >= 8:
                # Reversal entries: extreme Williams %R with trend and volume
                oversold = willr[i] < -80
                overbought = willr[i] > -20
                
                # Long: oversold in uptrend with volume surge
                if oversold and daily_trend_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: overbought in downtrend with volume surge
                elif overbought and daily_trend_aligned[i] == -1 and volume_filter:
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