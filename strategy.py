#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter
Hypothesis: Camarilla R1/S1 breakouts on 1h timeframe with 4h EMA50 trend filter and volume spike confirmation.
Only trade breakouts in direction of 4h trend with volume > 1.5x 20-period average.
Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-30 trades/year.
Works in both bull and bear markets via trend alignment and volume confirmation.
Breakouts represent strong momentum shifts with lower false signals when confirmed by volume and trend.
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
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA50 on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels on 4h data (using previous 4h bar)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = close, H = high, L = low of previous period
    camarilla_R1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12
    camarilla_S1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (standard 1-bar delay for completed bar)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R1_4h)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S1_4h)
    
    # Volume spike filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with trend and volume filters
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > camarilla_R1_aligned[i]) and (close[i] > ema50_4h_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_S1_aligned[i]) and (close[i] < ema50_4h_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below EMA50 (trend reversal) or opposite Camarilla level
            exit_signal = (close[i] < ema50_4h_aligned[i]) or (close[i] < camarilla_S1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above EMA50 (trend reversal) or opposite Camarilla level
            exit_signal = (close[i] > ema50_4h_aligned[i]) or (close[i] > camarilla_R1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0