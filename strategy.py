#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA20 trend filter and volume confirmation
# Uses 1h timeframe for precise entry timing, 4h for signal direction (trend and Camarilla levels)
# Volume spike confirms institutional interest. Discrete sizing (0.20) minimizes fee churn.
# Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years.
# Works in bull (breakouts with volume) and bear (trend continuation after pullbacks to EMA).

name = "1h_Camarilla_R3_S3_Breakout_4hEMA20_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA20 and Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) calculation
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 4h HTF data for Camarilla pivot calculation (using prior 4h bar's OHLC)
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formulas: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_high = close_4h + 1.1 * (high_4h - low_4h)  # R3
    camarilla_low = close_4h - 1.1 * (high_4h - low_4h)   # S3
    
    # Align Camarilla levels to 1h timeframe (using prior 4h bar's levels)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for EMA + 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Camarilla breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > camarilla_high_aligned[i-1]  # Break above R3
        breakout_down = curr_close < camarilla_low_aligned[i-1]  # Break below S3
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike[i]
        # Trend filter: price above/below 4h EMA20
        uptrend = curr_close > ema_20_4h_aligned[i]
        downtrend = curr_close < ema_20_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla breakout down, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakdown or trend reversal
            if curr_close < camarilla_low_aligned[i] or curr_close < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout or trend reversal
            if curr_close > camarilla_high_aligned[i] or curr_close > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals