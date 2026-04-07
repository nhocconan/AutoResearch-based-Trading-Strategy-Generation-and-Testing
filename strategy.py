#!/usr/bin/env python3
"""
12h_atr_breakout_1w_trend_volume_v2
Hypothesis: ATR breakout from weekly ATR-based channels combined with weekly EMA trend filter and volume confirmation.
In trending markets, price breaks out of volatility-based channels and continues in the direction of the weekly trend.
Uses weekly ATR channels for dynamic support/resistance, weekly EMA for trend filter, and volume spike for confirmation.
Designed for 12h timeframe to capture multi-day moves with low frequency (target: 15-30 trades/year) to minimize fee drag.
Works in both bull and bear markets by following the trend defined by higher timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_atr_breakout_1w_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for ATR channels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly ATR-based channels (using previous week's ATR)
    atr_shift = np.roll(atr_14, 1)
    atr_shift[0] = np.nan
    
    upper_channel = close_1w + 1.5 * atr_shift
    lower_channel = close_1w - 1.5 * atr_shift
    
    # Weekly EMA for trend filter
    ema_20 = close_1w.ewm(span=20, adjust=False).mean().values
    
    # Align all weekly data to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation (24-period average = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below lower channel or trend turns bearish
            if close[i] <= lower_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel or trend turns bullish
            if close[i] >= upper_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with volume and bullish trend
            if (close[i] > upper_aligned[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with volume and bearish trend
            elif (close[i] < lower_aligned[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals