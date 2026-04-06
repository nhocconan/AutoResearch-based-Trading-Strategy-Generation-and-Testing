#!/usr/bin/env python3
"""
6h Elder Ray Power with 1d Trend Filter and Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) captures institutional buying/selling pressure. 
Combined with 1d EMA50 trend filter for direction and volume confirmation to avoid false signals. 
Works in bull (buy when Bull Power > 0 and rising) and bear (sell when Bear Power > 0 and rising).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate EMA13 for Elder Ray (6-period EMA approximation for 6h timeframe)
    # Using 13-period EMA on 6h close
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2 + ema13[i-1] * 11) / 13
    
    # Elder Ray Power
    bull_power = np.full(n, np.nan)  # High - EMA13
    bear_power = np.full(n, np.nan)  # EMA13 - Low
    
    for i in range(n):
        if not np.isnan(ema13[i]):
            bull_power[i] = high[i] - ema13[i]
            bear_power[i] = ema13[i] - low[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for EMA13 and other indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.3x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.3
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power becomes positive and rising OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            bear_power_rising = i > start and bear_power[i] > bear_power[i-1]
            if (bear_power[i] > 0 and bear_power_rising) or \
               (trend_1d_aligned[i] == -1) or \
               (close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: Bull Power becomes positive and rising OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            bull_power_rising = i > start and bull_power[i] > bull_power[i-1]
            if (bull_power[i] > 0 and bull_power_rising) or \
               (trend_1d_aligned[i] == 1) or \
               (close[i] > entry_price + 2.0 * atr[i]):
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
                # Elder Ray signals with trend and volume filter
                bull_power_rising = i > start and bull_power[i] > bull_power[i-1]
                bear_power_rising = i > start and bear_power[i] > bear_power[i-1]
                
                # Long: Bull Power > 0 and rising with bullish 1d trend + volume
                if bull_power[i] > 0 and bull_power_rising and \
                   trend_1d_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: Bear Power > 0 and rising with bearish 1d trend + volume
                elif bear_power[i] > 0 and bear_power_rising and \
                     trend_1d_aligned[i] == -1 and volume_filter:
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