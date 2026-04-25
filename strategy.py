#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum; 1w EMA50 filters trend direction;
Volume spike confirms breakout strength; ATR stoploss limits drawdown.
Designed for 12h timeframe to avoid overtrading (target: 50-150 trades over 4 years).
Works in both bull and bear markets via trend filter and symmetric breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1w close for trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) on 12h for stoploss
    atr_14_12h = np.full(n, np.nan)
    for i in range(14, n):
        tr = np.max([
            high[i] - low[i],
            np.abs(high[i] - close[i-1]),
            np.abs(low[i] - close[i-1])
        ])
        if i == 14:
            atr_14_12h[i] = np.mean([
                high[1:15] - low[1:15],
                np.abs(high[1:15] - close[0:14]),
                np.abs(low[1:15] - close[0:14])
            ])
        else:
            atr_14_12h[i] = (atr_14_12h[i-1] * 13 + tr) / 14
    
    # Calculate Donchian(20) channels on 12h
    donchian_high_20 = np.full(n, np.nan)
    donchian_low_20 = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high_20[i] = np.max(high[i-19:i+1])
        donchian_low_20[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 12h volume confirmation
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Donchian, volume MA, ATR
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(vol_ma_20_12h[i]) or np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1w_aligned[i]
        upper_channel = donchian_high_20[i]
        lower_channel = donchian_low_20[i]
        vol_ma = vol_ma_20_12h[i]
        atr = atr_14_12h[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout entries
            # Long: price breaks above upper Donchian channel AND above 1w EMA50 AND volume confirmation
            long_entry = (curr_close > upper_channel and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below lower Donchian channel AND below 1w EMA50 AND volume confirmation
            short_entry = (curr_close < lower_channel and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below lower Donchian channel OR stoploss hit
            if curr_close < lower_channel or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stoploss: raise stop to break even + 0.5*ATR if in profit
                if curr_close > entry_price + atr:
                    atr_stop = max(atr_stop, entry_price + 0.5 * atr)
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian channel OR stoploss hit
            if curr_close > upper_channel or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stoploss: lower stop to break even - 0.5*ATR if in profit
                if curr_close < entry_price - atr:
                    atr_stop = min(atr_stop, entry_price - 0.5 * atr)
    
    return signals

name = "12h_Donchian20_1wEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0