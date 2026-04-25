#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with 1d EMA34 Trend and Volume Spike Filter
Hypothesis: Camarilla pivot levels (R3/S3) act as strong intraday support/resistance. 
Breakouts above R3 or below S3 with volume confirmation and aligned with 1d EMA34 trend 
capture momentum moves while filtering false breakouts in ranging markets. 
Discrete sizing (0.0, ±0.30) minimizes fee churn. Target: 20-40 trades/year on 4h.
Works in bull markets (long breakouts above R3 in uptrend) and bear markets 
(short breakdowns below S3 in downtrend) by requiring trend alignment.
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
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d average volume for volume spike filter
    avg_vol_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate Camarilla pivot levels for 4h timeframe using previous day's OHLC
    # We need to get previous day's high, low, close for each 4h bar
    # Since we're on 4h timeframe, we'll use the 1d data to compute daily pivots
    # and align them to 4h bars
    
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + ((high-low) * 1.1/2)
    # R3 = close + ((high-low) * 1.1/4)
    # S3 = close - ((high-low) * 1.1/4)
    # S4 = close - ((high-low) * 1.1/2)
    camarilla_r3 = daily_close + ((daily_high - daily_low) * 1.1 / 4)
    camarilla_s3 = daily_close - ((daily_high - daily_low) * 1.1 / 4)
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume calculations
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        avg_volume = avg_vol_1d_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        # Volume spike: current volume > 1.5 * average volume
        volume_spike = curr_volume > (1.5 * avg_volume)
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > r3_level) and volume_spike and (curr_close > ema_trend)
            # Short: price breaks below S3 AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < s3_level) and volume_spike and (curr_close < ema_trend)
            
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
            # Exit: price falls back below R3 OR price < 1d EMA34 (trend change)
            if (curr_close < r3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises back above S3 OR price > 1d EMA34 (trend change)
            if (curr_close > s3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0