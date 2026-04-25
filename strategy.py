#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 12h EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla pivot breakouts capture institutional order flow, 
12h EMA34 filters medium-term trend, volume spike confirms participation, 
and chop filter avoids ranging markets. Works in bull/bear via trend filter.
Target: 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # We need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3, R3, S3, S3 levels
    # R4 = Close + ((High-Low) * 1.1/2)
    # R3 = Close + ((High-Low) * 1.1/4)
    # S3 = Close - ((High-Low) * 1.1/4)
    # S4 = Close - ((High-Low) * 1.1/2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Choppiness Index filter (avoid ranging markets)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # We'll use a simplified version: high-low range relative to ATR
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    price_range_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values - pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / price_range_14) / np.log10(14)
    chop_filter = chop > 50  # Only trade when chop > 50 (not too trending, not too choppy)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 30, 14) + 1  # +1 for previous bar reference
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        
        # Camarilla breakout conditions
        breakout_long = curr_high > camarilla_r3_aligned[i]  # Break above R3
        breakout_short = curr_low < camarilla_s3_aligned[i]  # Break below S3
        
        # Trend filter: price above/below 12h EMA34
        uptrend = curr_close > ema_34_12h_aligned[i]
        downtrend = curr_close < ema_34_12h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend alignment + volume + chop filter
            long_entry = breakout_long and uptrend and vol_spike and chop_ok
            short_entry = breakout_short and downtrend and vol_spike and chop_ok
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price retouches S3 level OR trend reverses
            if curr_close < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches R3 level OR trend reverses
            if curr_close > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0