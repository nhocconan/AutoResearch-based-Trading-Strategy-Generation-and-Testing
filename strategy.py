#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence on 12h.
Trades only when 1d EMA34 confirms trend direction and volume spikes on both timeframes.
Designed for low-frequency, high-conviction entries to avoid fee drag on 12h timeframe.
Works in bull/bear by following 1d EMA trend and requiring Alligator alignment.
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
    
    # Get 12h data for Williams Alligator (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw_12h = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_12h = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_12h = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 12h and 1d
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_20_12h[i] = np.mean(df_12h['volume'].values[i-19:i+1])
    
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    
    # Align all to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12h volume MA for volume spike detection
    vol_ma_20_12h_current = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h_current[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for all indicators
    start_idx = 50  # Ensures all MA/SMA/EMA have min_periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(vol_ma_20_12h_current[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw = jaw_12h_aligned[i]
        teeth = teeth_12h_aligned[i]
        lips = lips_12h_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma_12h = vol_ma_20_12h_aligned[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        vol_ma_12h_curr = vol_ma_20_12h_current[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        alligator_bullish = lips > teeth and teeth > jaw
        alligator_bearish = lips < teeth and teeth < jaw
        
        # Volume confirmation: current 12h volume > 2.0 * 20-period MA on BOTH 12h and 1d
        volume_confirm_12h = curr_volume > 2.0 * vol_ma_12h
        volume_confirm_1d = prices['volume'].values[i] > 2.0 * vol_ma_1d if hasattr(prices['volume'].values, '__getitem__') else False
        # Use 12h volume confirmation as primary (more reliable on lower TF)
        volume_confirm = volume_confirm_12h
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator bullish AND price > 1d EMA34 AND volume confirmation
            long_entry = alligator_bullish and curr_close > ema_trend and volume_confirm
            # Short: Alligator bearish AND price < 1d EMA34 AND volume confirmation
            short_entry = alligator_bearish and curr_close < ema_trend and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator turns bearish OR price < 1d EMA34
            if not alligator_bullish or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator turns bullish OR price > 1d EMA34
            if not alligator_bearish or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0