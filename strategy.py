#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Daily Camarilla H3/L3 levels act as strong intraday support/resistance on 4h charts.
Breakouts above H3 or below L3 with volume confirmation and aligned daily EMA34 trend capture
continuation moves. The daily EMA34 filter ensures we trade with higher timeframe momentum,
reducing false breakouts. Volume spike confirms institutional participation. Designed for
low trade frequency (20-50/year) to minimize fee drag on 4h timeframe. Works in both bull
and bear markets by following the daily trend direction.
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
    
    # Get daily data for EMA34 trend and Camarilla pivot (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on daily close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get daily OHLC values for Camarilla calculation
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Daily Camarilla pivot levels
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    daily_h3 = d_close + (d_high - d_low) * 1.1 / 4
    daily_l3 = d_close - (d_high - d_low) * 1.1 / 4
    
    # Align Daily Camarilla levels to 4h timeframe
    daily_h3_aligned = align_htf_to_ltf(prices, df_1d, daily_h3)
    daily_l3_aligned = align_htf_to_ltf(prices, df_1d, daily_l3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and daily pivots
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(daily_h3_aligned[i]) or 
            np.isnan(daily_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        h3_level = daily_h3_aligned[i]
        l3_level = daily_l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 level AND volume spike AND price > daily EMA34 (uptrend)
            long_entry = (curr_close > h3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 level AND volume spike AND price < daily EMA34 (downtrend)
            short_entry = (curr_close < l3_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 level (reversal) OR price crosses below EMA (trend change)
            if (curr_close < l3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 level (reversal) OR price crosses above EMA (trend change)
            if (curr_close > h3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0