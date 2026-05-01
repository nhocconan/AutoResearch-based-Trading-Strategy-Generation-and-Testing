#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation
# Primary timeframe 1h for precise entry timing, HTF 4h for trend alignment and 1d for Camarilla levels.
# Camarilla R3/S3 represent stronger support/resistance than R1/S1, reducing false breakouts.
# Volume spike confirms institutional participation. 4h EMA34 ensures medium-term trend alignment.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Designed for low trade frequency (target: 15-37 trades/year) to minimize fee drag in 1h timeframe.
# Works in both bull and bear markets by taking breakouts in direction of 4h trend.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # We use R3 and S3 for stronger breakout signals
    # R3 = Close + 1.1*(High-Low)/4
    # S3 = Close - 1.1*(High-Low)/4
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 4
    s3 = close_1d - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA34
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3, volume spike, uptrend
            if close[i] > r3_aligned[i] and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: break below S3, volume spike, downtrend
            elif close[i] < s3_aligned[i] and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below R3 or trend reversal
            if close[i] < r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on break above S3 or trend reversal
            if close[i] > s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals