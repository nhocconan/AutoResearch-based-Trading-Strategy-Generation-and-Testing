#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + 1d EMA34 Trend Filter + Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance where breakouts capture institutional flow. 
1d EMA34 filters for primary trend alignment to avoid counter-trend trades. Volume spike confirms participation. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year on 4h.
Works in bull/bear via trend filter - only takes longs in uptrend, shorts in downtrend.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3
    # For intraday, we need previous day's typical price
    # We'll use daily resampling concept but via HTF data
    # Since we have 1d data, we can calculate proper Camarilla levels
    
    # For 4h timeframe, we need to calculate Camarilla levels based on previous 1d OHLC
    # We'll shift the 1d data by 1 to get previous day's levels
    if len(df_1d) >= 2:
        prev_day_high = df_1d['high'].shift(1).values
        prev_day_low = df_1d['low'].shift(1).values
        prev_day_close = df_1d['close'].shift(1).values
        
        # Calculate pivot point
        pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
        # Calculate Camarilla levels
        range_val = prev_day_high - prev_day_low
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        
        # Align to 4h timeframe
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    else:
        # Not enough data, return zeros
        return np.zeros(n)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Camarilla breakout conditions
        breakout_long = curr_high > r3_aligned[i]  # Break above R3
        breakout_short = curr_low < s3_aligned[i]  # Break below S3
        
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend alignment + volume
            long_entry = breakout_long and uptrend and vol_spike
            short_entry = breakout_short and downtrend and vol_spike
            
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
            # Exit: price retouches pivot level OR trend reverses
            if curr_close < pivot[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches pivot level OR trend reverses
            if curr_close > pivot[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0