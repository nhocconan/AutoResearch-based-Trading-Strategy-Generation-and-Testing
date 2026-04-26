#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeConfirmation
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume confirmation capture institutional moves while minimizing fee drag. 
Uses 4h for signal direction (trend and Camarilla levels) and 1h only for entry timing precision. 
Targets 15-35 trades/year via strict volume confirmation (2.5x average) and session filter (08-20 UTC). 
Position size fixed at 0.20 to manage drawdown. Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes during low volatility).
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (UTC 8-20)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 4h bar (H1, L1, C1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    if len(high_4h) < 2:
        camarilla_r1 = np.full_like(close_4h_arr, np.nan)
        camarilla_s1 = np.full_like(close_4h_arr, np.nan)
    else:
        camarilla_r1 = close_4h_arr[:-1] + 1.1 * (high_4h[:-1] - low_4h[:-1]) / 12
        camarilla_s1 = close_4h_arr[:-1] - 1.1 * (high_4h[:-1] - low_4h[:-1]) / 12
        camarilla_r1 = np.concatenate([[np.nan], camarilla_r1])
        camarilla_s1 = np.concatenate([[np.nan], camarilla_s1])
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume average (24-period = 1 day on 1h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(24, 50)  # volume MA, 4h EMA
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get aligned values
        ema_50_4h_val = ema_50_4h_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2.5x 24-period average (strict)
        volume_confirmed = vol_val > 2.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above R1 with uptrend (close > EMA50) and volume confirmation
            long_signal = (high_val > r1_val) and (close_val > ema_50_4h_val) and volume_confirmed
            # Short: price breaks below S1 with downtrend (close < EMA50) and volume confirmation
            short_signal = (low_val < s1_val) and (close_val < ema_50_4h_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S1 (exit long)
            if low_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA50
            elif close_val < ema_50_4h_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            if high_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA50
            elif close_val > ema_50_4h_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0