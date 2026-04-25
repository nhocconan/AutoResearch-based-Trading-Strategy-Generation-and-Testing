#!/usr/bin/env python3
"""
6h Camarilla Pivot H3/L3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance on 1d timeframe.
Breakout above H3 or below L3 with 1d EMA34 trend alignment and volume spike captures
continuation moves. Works in both bull/bear markets by following the 1d trend.
Uses 6h timeframe with 1d HTF for trend/targets. Targets 50-150 total trades over 4 years.
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
    
    # Get 1d data for Camarilla pivots, EMA34 trend, and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # H3 = Close + 1.1*(High - Low)/2
    # L3 = Close - 1.1*(High - Low)/2
    # Using previous day's values to avoid look-ahead
    h3 = df_1d['close'].values + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 2
    l3 = df_1d['close'].values - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 2
    
    # Calculate 1d EMA34 for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    
    # Calculate 1d volume MA20 for volume confirmation
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    
    # Align all 1d indicators to 6h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        
        # Volume confirmation: current 6h volume > 2.0 * 20-period 1d average
        # Scale volume comparison: 6h volume vs daily average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_1d / 4  # 4x 6h bars in 1d
        
        if position == 0:
            # Look for entry signals
            # Long: Break above H3 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > h3_val and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below L3 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < l3_val and 
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
            # Exit: Price falls below EMA34 OR breaks below L3 (failed breakout)
            if (curr_close < ema_trend or curr_low < l3_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price rises above EMA34 OR breaks above H3 (failed breakout)
            if (curr_close > ema_trend or curr_high > h3_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0