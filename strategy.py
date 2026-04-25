#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_WeeklyTrend_VolumeConfirm
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume spike confirmation.
In weekly uptrend (price > weekly EMA50), trade breakouts in direction of trend; in weekly downtrend,
fade at Camarilla extremes. Uses discrete sizing (0.25) to minimize fee churn. Target: 7-25 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at extremes when trend weakens.
Weekly EMA50 provides stable trend filter for daily timeframe, reducing whipsaw.
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
    
    # Get 1d data for Camarilla calculations (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous bar)
    R1_1d = np.full(len(close_1d), np.nan)
    S1_1d = np.full(len(close_1d), np.nan)
    R2_1d = np.full(len(close_1d), np.nan)
    S2_1d = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Camarilla levels based on previous 1d bar's range
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R2_1d[i] = close_prev + (range_prev * 1.1 / 2)  # R2 level
            S2_1d[i] = close_prev - (range_prev * 1.1 / 2)  # S2 level
            R1_1d[i] = close_prev + (range_prev * 1.1 / 4)  # R1 level
            S1_1d[i] = close_prev - (range_prev * 1.1 / 4)  # S1 level
    
    # Align Camarilla levels to original timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Weekly uptrend regime
                # Long: break above R1 with volume spike
                long_signal = (close[i] > R1_1d_aligned[i]) and vol_spike[i]
                # Short: break below S1 only if strong volume spike (counter-trend fade)
                short_signal = (close[i] < S1_1d_aligned[i]) and vol_spike[i] and (close[i] < ema_trend * 0.98)
            else:  # Weekly downtrend regime
                # Short: break below S1 with volume spike
                short_signal = (close[i] < S1_1d_aligned[i]) and vol_spike[i]
                # Long: break above R1 only if strong volume spike (counter-trend fade)
                long_signal = (close[i] > R1_1d_aligned[i]) and vol_spike[i] and (close[i] > ema_trend * 1.02)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: touch S1 or trend reversal
            exit_signal = (close[i] < S1_1d_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: touch R1 or trend reversal
            exit_signal = (close[i] > R1_1d_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0