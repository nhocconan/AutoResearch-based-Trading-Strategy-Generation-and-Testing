#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout with 12h EMA34 Trend and Volume Spike Filter
Hypothesis: Camarilla H3/L3 levels provide strong intraday support/resistance. Breakouts above H3 or below L3 with 
volume confirmation and aligned 12h EMA34 trend capture momentum moves while avoiding false breakouts. The 12h EMA34 
ensures we trade with higher timeframe momentum, reducing whipsaws. Volume spike confirms institutional participation. 
Designed for low-moderate trade frequency (19-50/year) on 4h timeframe to work in both bull and bear markets via 
trend following with strict entry conditions.
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
    
    # Get 12h data for EMA34 trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 12h close for trend
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 12h data for Camarilla pivot calculation (H3, L3 levels)
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each 12h bar: based on previous 12h bar's high, low, close
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Camarilla formulas:
    # H3 = close + (high - low) * 1.1/2
    # L3 = close - (high - low) * 1.1/2
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to LTF (4h) - no extra delay needed as pivots are based on completed 12h bar
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and to avoid NaN from shift
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_12h_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 resistance AND volume spike AND price > 12h EMA34 (uptrend)
            long_entry = (curr_close > h3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 support AND volume spike AND price < 12h EMA34 (downtrend)
            short_entry = (curr_close < l3_level) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below L3 support (broken support) OR price crosses below EMA (trend change)
            if (curr_close < l3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 resistance (broken resistance) OR price crosses above EMA (trend change)
            if (curr_close > h3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0