#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, Camarilla pivot R1/S1 breakouts with 4h EMA50 trend filter and volume confirmation (>1.5x 24-bar avg) capture institutional breakouts in both bull and bear markets. Uses 4h for signal direction to reduce noise and avoid lower timeframe whipsaws, 1h only for precise entry timing. Session filter (08-20 UTC) reduces off-hours noise. Target: 15-35 trades/year to minimize fee drag while maintaining edge via trend filter and volume confirmation.
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
    open_time = prices['open_time'].values
    
    # Get 4h data for HTF trend and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots on 4h (using previous 4h bar)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous 4h bar's H, L, C to calculate pivots for current 4h bar
    # So we shift the 4h data by 1 bar
    if len(high_4h) < 2:
        return np.zeros(n)
    
    # Previous 4h bar's OHLC
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    # First bar has no previous, set to NaN
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    # Calculate Camarilla R1 and S1 for each 4h bar
    camarilla_r1_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12
    camarilla_s1_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12
    
    # Align to 1h timeframe (with proper delay for completed 4h bar)
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Volume average (24-period = 1 day on 1h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, 24)  # EMA50 lookback, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_4h_aligned[i]) or 
            np.isnan(camarilla_s1_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            not in_session[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get aligned values
        ema_50_val = ema_50_4h_aligned[i]
        camarilla_r1_val = camarilla_r1_4h_aligned[i]
        camarilla_s1_val = camarilla_s1_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with uptrend (close > EMA50) and volume confirmation
            long_signal = (close_val > camarilla_r1_val) and (close_val > ema_50_val) and volume_confirmed
            # Short: price breaks below Camarilla S1 with downtrend (close < EMA50) and volume confirmation
            short_signal = (close_val < camarilla_s1_val) and (close_val < ema_50_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions:
            # 1. Price breaks below Camarilla S1 (opposite level)
            if close_val < camarilla_s1_val:
                signals[i] = 0.0
                position = 0
            # 2. Trend reversal: close crosses below EMA50
            elif close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions:
            # 1. Price breaks above Camarilla R1 (opposite level)
            if close_val > camarilla_r1_val:
                signals[i] = 0.0
                position = 0
            # 2. Trend reversal: close crosses above EMA50
            elif close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0