#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong support/resistance. Breakout with 1d uptrend/downtrend and volume spike captures institutional flow. Works in bull/bear via trend filter. Targets 75-200 trades over 4 years.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (need HLC of completed 1d)
    # We'll compute daily HLC and shift by 1 to avoid look-ahead
    df_1d = df_1d.copy()
    df_1d['typical'] = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually standard: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # Using R3/S3 as breakout levels
    df_1d['camarilla_R3'] = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    df_1d['camarilla_S3'] = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    
    # Align to 4h: need previous day's levels (so shift by 1)
    camarilla_R3 = df_1d['camarilla_R3'].values
    camarilla_S3 = df_1d['camarilla_S3'].values
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Calculate 20-period volume MA for 4h volume confirmation
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(34, 20)  # 34 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        camarilla_R3 = camarilla_R3_aligned[i]
        camarilla_S3 = camarilla_S3_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: close breaks above camarilla R3 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_close > camarilla_R3 and 
                         curr_close > ema_trend and volume_confirm)
            # Short: close breaks below camarilla S3 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_close < camarilla_S3 and 
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
            # Exit: close falls below camarilla S3 OR price falls below EMA34
            if (curr_close < camarilla_S3 or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: close rises above camarilla R3 OR price rises above EMA34
            if (curr_close > camarilla_R3 or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0