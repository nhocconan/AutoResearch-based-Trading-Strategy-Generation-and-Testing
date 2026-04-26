#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter_v1
Hypothesis: Trade 12h breakouts of Camarilla R1/S1 levels filtered by 1d EMA trend and volume confirmation.
Camarilla pivot levels provide precise intraday support/resistance. In strong 1d trends,
breakouts of these levels have high continuation probability. Volume confirmation ensures
institutional participation. Designed for 12h timeframe with low trade frequency (12-37/year)
to minimize fee drag. Works in both bull and bear markets by following 1d trend.
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
    open_time = prices['open_time']
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate previous day's Camarilla levels (R1, S1) from 1d data
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align HTF indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 1.5x median volume (using 24 periods = 12h * 2)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (34), volume median (24)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long: price breaks above R1, uptrend (close > 1d EMA34), volume confirmation
            long_signal = (close_val > camarilla_r1_val) and \
                          (close_val > ema_34_1d_val) and \
                          (volume_val > 1.5 * vol_median_val)
            # Short: price breaks below S1, downtrend (close < 1d EMA34), volume confirmation
            short_signal = (close_val < camarilla_s1_val) and \
                           (close_val < ema_34_1d_val) and \
                           (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Exit: trend reversal (close < 1d EMA34) or price retracement to EMA after minimum holding
            if bars_since_entry >= 4 and ((close_val < ema_34_1d_val) or 
                                          (abs(close_val - ema_34_1d_val) < 0.5 * abs(high[i] - low[i]))):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Exit: trend reversal (close > 1d EMA34) or price retracement to EMA after minimum holding
            if bars_since_entry >= 4 and ((close_val > ema_34_1d_val) or 
                                          (abs(close_val - ema_34_1d_val) < 0.5 * abs(high[i] - low[i]))):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0