#!/usr/bin/env python3
"""
4h Camarilla R3 S3 Breakout with Daily EMA Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong intraday support/resistance.
Breakouts above R3 or below S3 with daily EMA trend alignment and volume spikes capture
strong trending moves. Uses 4h timeframe with 1d HTF for trend and pivot calculation.
Targeting 75-200 trades over 4 years (19-50/year) to balance opportunity with fee drag.
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
    
    # Get 1d data for Camarilla pivots and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3 and S3 as key levels
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    
    # Camarilla levels need no extra delay - they are based on completed 1d bar
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate 34-period EMA on 1d close (only needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for 4h volume spike
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 (resistance), above 1d EMA, volume confirmation
            long_entry = (curr_close > r3_level and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below S3 (support), below 1d EMA, volume confirmation
            short_entry = (curr_close < s3_level and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below S3 (support) OR below 1d EMA
            if curr_close < s3_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above R3 (resistance) OR above 1d EMA
            if curr_close > r3_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0