#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla H3 (resistance) and L3 (support) levels from the prior 1d
act as significant intraday support/resistance. Breakouts above H3 or below L3
with volume confirmation and aligned 1d EMA34 trend capture institutional moves.
The 1d EMA34 ensures we trade with higher timeframe momentum, reducing false
breakouts. Volume spike confirms participation. Designed for low trade frequency
(12-37/year) on 12h timeframe to minimize fee drag and improve test generalization.
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
    
    # Get 1d data for EMA34 trend and Camarilla levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d OHLC (use shift(1) for completed bar)
    # Camarilla levels: H3, L3, H4, L4
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    camarilla_base = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    camarilla_range = prev_day_high - prev_day_low
    camarilla_h3 = camarilla_base + camarilla_range * 1.1 / 4.0
    camarilla_l3 = camarilla_base - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and prior day data
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla H3 (strong resistance) AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > camarilla_h3) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Camarilla L3 (strong support) AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < camarilla_l3) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below Camarilla L3 (support) OR price crosses below EMA (trend change)
            if (curr_close < camarilla_l3) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Camarilla H3 (resistance) OR price crosses above EMA (trend change)
            if (curr_close > camarilla_h3) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0