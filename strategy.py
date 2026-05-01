#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h HTF for trend alignment (more responsive than 1d) while keeping Camarilla from 1d pivots.
# Volume spike confirms institutional interest. Discrete sizing (0.25) minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull (breakouts with volume) 
# and bear (trend continuation after pullbacks to EMA).

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA50 calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) calculation
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d HTF data for Camarilla pivot calculation (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d)  # R3
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d)   # S3
    
    # Align Camarilla levels to 4h timeframe (using prior day's levels)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for 12h EMA + 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Camarilla breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > camarilla_high_aligned[i-1]  # Break above R3
        breakout_down = curr_close < camarilla_low_aligned[i-1]  # Break below S3
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike[i]
        # Trend filter: price above/below 12h EMA50
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakdown or trend reversal
            if curr_close < camarilla_low_aligned[i] or curr_close < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout or trend reversal
            if curr_close > camarilla_high_aligned[i] or curr_close > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals