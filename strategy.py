#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout + 4h EMA20 Trend + Volume Spike
Hypothesis: Camarilla H3 (resistance) and L3 (support) levels from 4h pivots act as 
institutional order flow zones. Breaking above H3 with volume and 4h uptrend signals 
bullish momentum; breaking below L3 with volume and 4h downtrend signals bearish 
momentum. Works in bull/bear markets by only taking breakouts aligned with 4h EMA20 
trend. Uses 1h timeframe for precise entry timing while using 4h for signal direction 
to minimize trades (target: 15-37/year) and reduce fee drag. Session filter (08-20 UTC) 
avoids low-volume Asian session noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 4h data for EMA20 trend filter and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h Camarilla pivot levels
    # Typical price = (high + low + close) / 3
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    # Camarilla pivot = typical price
    pivot = typical_price.values
    # Ranges
    hl_range = df_4h['high'].values - df_4h['low'].values
    # H3 = pivot + (high - low) * 1.1 / 4
    # L3 = pivot - (high - low) * 1.1 / 4
    camarilla_h3 = pivot + hl_range * 1.1 / 4
    camarilla_l3 = pivot - hl_range * 1.1 / 4
    
    # Align HTF levels to LTF with proper delay
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Pre-compute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA20 warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_20_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_20_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above H3 (resistance) AND above 4h EMA20 (uptrend filter)
            long_condition = (curr_close > h3_level) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below L3 (support) AND below 4h EMA20 (downtrend filter)
            short_condition = (curr_close < l3_level) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to L3 or trend breaks
            if curr_close <= l3_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to H3 or trend breaks
            if curr_close >= h3_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA20_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0