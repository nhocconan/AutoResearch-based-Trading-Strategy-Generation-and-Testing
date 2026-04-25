#!/usr/bin/env python3
"""
1h_4hCamarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: On 1h timeframe, price breaking above/below 4h Camarilla H3/L3 levels with volume spike,
aligned with 1d EMA34 trend, captures momentum in both bull and bear markets.
Uses 4h/1d for signal direction, 1h only for entry timing.
Session filter (08-20 UTC) reduces noise trades.
Target: 15-37 trades/year per symbol.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h (based on previous 4h bar's high/low/close)
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    # Using previous 4h bar's values to avoid look-ahead
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    camarilla_h3_4h = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 2.0
    camarilla_l3_4h = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 2.0
    
    # Align Camarilla levels to 1h timeframe
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session or data not ready
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h3 = h3_4h_aligned[i]
        l3 = l3_4h_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_close > h3) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_close < l3) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 OR price crosses below EMA
            if (curr_close < l3) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 OR price crosses above EMA
            if (curr_close > h3) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hCamarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0