#!/usr/bin/env python3
"""
12h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) from 1d act as intraday support/resistance.
Breakouts above R1 or below S1 with 1d EMA34 trend alignment and volume spike
capture institutional moves. Works in bull/bear via trend filter + volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Camarilla pivots and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from 1d OHLC
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 12h timeframe (no extra delay - levels known at 1d close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(close_1d).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for 12h volume confirmation
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(34, 20)  # 34 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > r1_level and 
                         curr_close > ema_trend and volume_confirm)
            # Short: price breaks below S1 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < s1_level and 
                          curr_close < ema_trend and volume_confirm)
            
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
            # Exit: price falls below S1 (reversal) OR price falls below EMA34 (trend change)
            if (curr_low < s1_level or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above R1 (reversal) OR price rises above EMA34 (trend change)
            if (curr_high > r1_level or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0