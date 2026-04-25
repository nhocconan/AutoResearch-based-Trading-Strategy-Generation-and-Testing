#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 1h timeframe with 4h EMA20 trend filter and volume spike confirmation. 
Only trade breakouts in direction of 4h trend with volume > 1.5x 20-period average. Uses discrete position sizing (0.20) 
to minimize fee churn. Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag. Designed to work in 
both bull and bear markets via 4h trend alignment and volume confirmation to filter false breakouts.
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
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA20 on 4h close for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels on 4h data (based on previous day's OHLC)
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need to calculate these for each 4h bar using that bar's OHLC
    camarilla_r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12
    camarilla_s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align HTF indicators to 1h timeframe (standard 1-bar delay for completed bar)
    ema20_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h, additional_delay_bars=1)
    
    # Calculate volume spike on 1h: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA20 (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema20_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with trend and volume filters
            # Long: price breaks above R1 in uptrend (close > EMA20) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA20) with volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema20_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema20_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below EMA20 (trend reversal) or opposite Camarilla level (S1)
            exit_signal = (close[i] < ema20_aligned[i]) or (close[i] < camarilla_s1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above EMA20 (trend reversal) or opposite Camarilla level (R1)
            exit_signal = (close[i] > ema20_aligned[i]) or (close[i] > camarilla_r1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0