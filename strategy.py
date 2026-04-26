#!/usr/bin/env python3
"""
6h_WeeklyDonchian_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: Weekly Donchian breakout (20-week high/low) with 1d EMA50 trend filter and volume confirmation.
In bull markets, price makes new weekly highs; in bear markets, new weekly lows. The 1d EMA50 ensures we only
trade in the direction of the daily trend to avoid counter-trend breakouts. Volume confirmation filters out
false breakouts. Designed for low trade frequency (~20-50/year) to minimize fee drag on 6h timeframe.
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
    
    # Get weekly data for Donchian channels (20-week lookback)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian: highest high and lowest low over past 20 weeks
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for 20 periods
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    highest_high_20w = rolling_max(high_1w, 20)
    lowest_low_20w = rolling_min(low_1w, 20)
    
    # Align weekly Donchian levels to 6h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20w)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20w)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.5x average volume (moderate to balance signal quality)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*6h = 6d lookback
    
    # ATR for stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of weekly Donchian (20), 1d EMA (50), volume MA (24), ATR (14)
    start_idx = max(20, 50, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        highest_high_val = highest_high_aligned[i]
        lowest_low_val = lowest_low_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above 20-week high, uptrend (close > 1d EMA50), volume confirmation
            long_signal = (high_val > highest_high_val) and (close_val > ema_50_1d_val) and (volume_val > 1.5 * vol_ma_val)
            # Short: break below 20-week low, downtrend (close < 1d EMA50), volume confirmation
            short_signal = (low_val < lowest_low_val) and (close_val < ema_50_1d_val) and (volume_val > 1.5 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_val  # wider stop for 6h volatility
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_val)
            # Exit: trailing stop hit or trend reversal (price < 1d EMA50)
            if (low_val < long_stop) or (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_val)
            # Exit: trailing stop hit or trend reversal (price > 1d EMA50)
            if (high_val > short_stop) or (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0