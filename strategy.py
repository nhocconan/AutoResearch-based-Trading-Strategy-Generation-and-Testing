#!/usr/bin/env python3
"""
12h Camarilla R3S3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla pivot breakouts on 12h timeframe capture institutional order flow, 
1d EMA34 filters long-term trend, volume spike confirms participation, and chop filter 
avoids extreme ranging/choppy markets. Works in bull/bear via trend filter. Target: 12-37 trades/year on 12h.
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
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (more conservative)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index filter (avoid ranging markets) - simplified version
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Handle first value where roll gives same index
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    price_range_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values - pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop = np.zeros_like(close)
    mask = price_range_14 > 0
    chop[mask] = 100 * np.log10(atr_14[mask] * 14 / price_range_14[mask]) / np.log10(14)
    # Filter: trade when market is not too choppy (CHOP < 61.8) and not too trending (CHOP > 38.2)
    chop_filter = (chop > 38.2) & (chop < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 24, 14) + 1  # +1 for previous bar reference
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
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
        
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
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

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0