#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
Only trade breakouts in direction of 12h trend. Uses discrete sizing (0.30) to minimize fee churn.
Designed for low trade frequency (~20-40/year) to work in both bull and bear markets via trend alignment.
Volume spike confirms institutional interest, reducing false breakouts in choppy markets.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Get 1d data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today based on yesterday's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = yesterday's close, H = yesterday's high, L = yesterday's low
    prev_close = close_1d[:-1]  # yesterday's close
    prev_high = high_1d[:-1]    # yesterday's high
    prev_low = low_1d[:-1]      # yesterday's low
    
    # Calculate R1 and S1 for each day
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (standard 1-bar delay for daily)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    
    # Volume spike detection on 4h: volume > 1.5 * 20-period average
    vol_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values  # using close as placeholder for volume series
    vol_ma_actual = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_actual * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with trend and volume filters
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and \
                         (close[i] > ema50_12h_aligned[i]) and \
                         volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and \
                          (close[i] < ema50_12h_aligned[i]) and \
                          volume_spike[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit when price moves back below R1 or trend reverses
            exit_signal = (close[i] < camarilla_r1_aligned[i]) or (close[i] < ema50_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit when price moves back above S1 or trend reverses
            exit_signal = (close[i] > camarilla_s1_aligned[i]) or (close[i] > ema50_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0