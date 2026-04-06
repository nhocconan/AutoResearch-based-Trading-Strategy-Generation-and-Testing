#!/usr/bin/env python3
"""
6h Williams %R with 1d Trend Filter and Volume Confirmation
Hypothesis: Williams %R identifies overbought/oversold conditions. Combined with daily trend filter (price vs EMA50) and volume surge (>1.5x 20-period average), it captures mean-reversion entries aligned with higher timeframe momentum. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend). Target: 80-180 total trades over 4 years (20-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williamsr_1d_trend_vol_v1"
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
    
    # Williams %R (14-period) on 6h data
    willr = np.full(n, np.nan)
    if n >= 14:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        for i in range(14, n):
            highest_high[i] = np.max(high[i-14:i])
            lowest_low[i] = np.min(low[i-14:i])
            denom = highest_high[i] - lowest_low[i]
            if denom != 0:
                willr[i] = (highest_high[i] - close[i]) / denom * -100
    
    # Get 1d data for trend filter (EMA50) and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA50 on daily close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # Daily trend: above EMA50 = bullish, below = bearish
    daily_trend = np.where(close_1d > ema_1d, 1, -1)
    
    # 20-period average volume on daily
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align daily trend and volume MA to 6h timeframe
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for Williams %R and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(willr[i]) or np.isnan(atr[i]) or 
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
            # Exit: Williams %R > -20 (overbought) OR against daily trend
            # Stoploss: price drops 2*ATR below entry
            if (willr[i] > -20 or
                daily_trend_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: Williams %R < -80 (oversold) OR against daily trend
            # Stoploss: price rises 2*ATR above entry
            if (willr[i] < -80 or
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
                # Mean reversion entries: oversold in uptrend, overbought in downtrend
                oversold = willr[i] < -80
                overbought = willr[i] > -20
                
                # Long: oversold with bullish daily trend + volume
                if oversold and daily_trend_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: overbought with bearish daily trend + volume
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