#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_Filter
Hypothesis: On 1h timeframe, trade Camarilla R1/S1 breakouts only when aligned with 4h trend (price > 4h EMA200 for longs, < 4h EMA200 for shorts). Volume confirmation (>1.8x 20-bar mean) ensures breakout conviction. Uses discrete sizing (0.20) and session filter (08-20 UTC) to reduce noise. Designed for 15-30 trades/year per symbol, effective in bull markets via breakouts and bear markets via trend-following shorts.
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
    
    # Pre-compute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA(200) on 4h for trend filter
    ema_200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align EMA200 to 1h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of prior bar)
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h)  # R1 = C + 1.1*(H-L)
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h)  # S1 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 1h timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: current volume > 1.8x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not in_session[i] or \
           np.isnan(ema_200_aligned[i]) or \
           np.isnan(camarilla_r1_aligned[i]) or \
           np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(vol_mean_20[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend (price > 4h EMA200) with volume confirmation
            # Short: price breaks below Camarilla S1 in downtrend (price < 4h EMA200) with volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_200_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_200_aligned[i]) and vol_confirm[i]
            
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
            # Exit when price moves back below 4h EMA200 (trend reversal)
            exit_signal = close[i] < ema_200_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above 4h EMA200 (trend reversal)
            exit_signal = close[i] > ema_200_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0