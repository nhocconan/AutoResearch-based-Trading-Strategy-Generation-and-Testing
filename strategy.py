#!/usr/bin/env python3
"""
1d Camarilla H3L3 Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Daily Camarilla H3/L3 levels act as significant intraday support/resistance. 
Breakouts above H3 or below L3 with volume confirmation and aligned 1w EMA34 trend 
capture institutional moves. The 1w EMA34 ensures we trade with higher timeframe 
momentum, reducing false breakouts. Volume spike confirms participation. 
Designed for low trade frequency (7-25/year) on 1d timeframe.
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
    
    # Get 1d data for Camarilla levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Use shift(1) to ensure we only use completed daily bars (no look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels calculation
    cam_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    cam_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (completed 1d bar only)
    cam_h3_aligned = align_htf_to_ltf(prices, df_1d, cam_h3)
    cam_l3_aligned = align_htf_to_ltf(prices, df_1d, cam_l3)
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and prior day data
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(cam_h3_aligned[i]) or 
            np.isnan(cam_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        cam_h3 = cam_h3_aligned[i]
        cam_l3 = cam_l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla H3 (strong resistance) AND volume spike AND price > 1w EMA34 (uptrend)
            long_entry = (curr_close > cam_h3) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Camarilla L3 (strong support) AND volume spike AND price < 1w EMA34 (downtrend)
            short_entry = (curr_close < cam_l3) and vol_spike and (curr_close < ema_trend)
            
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
            if (curr_close < cam_l3) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Camarilla H3 (resistance) OR price crosses above EMA (trend change)
            if (curr_close > cam_h3) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0