#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (price > 4h EMA50 for long, < 4h EMA50 for short) and 1d volume confirmation (>2.0x 20-bar mean volume). Uses 4h for signal direction and 1h for entry timing precision. Targets 15-30 trades/year per symbol by requiring confluence of HTF trend, LT breakout, and volume spike. Designed to work in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend) via strict entry conditions and session filter (08-20 UTC) to avoid low-liquidity hours.
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_mean_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = align_htf_to_ltf(prices, df_1d, volume_1d) > (vol_mean_20_1d * 2.0)
    
    # Calculate Camarilla levels from previous 1h bar (HLC of prior bar)
    camarilla_r1 = close + 1.1 * (high - low)  # R1 = C + 1.1*(H-L)
    camarilla_s1 = close - 1.1 * (high - low)  # S1 = C - 1.1*(H-L)
    
    # Use previous bar's levels to avoid look-ahead
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if not in session or data not ready
        if not in_session[i] or \
           np.isnan(ema_4h_aligned[i]) or \
           np.isnan(camarilla_r1_prev[i]) or \
           np.isnan(camarilla_s1_prev[i]) or \
           np.isnan(vol_spike_1d[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend (price > 4h EMA50) with volume confirmation and session
            long_signal = (close[i] > camarilla_r1_prev[i]) and (close[i] > ema_4h_aligned[i]) and vol_spike_1d[i]
            # Short: price breaks below Camarilla S1 in downtrend (price < 4h EMA50) with volume confirmation and session
            short_signal = (close[i] < camarilla_s1_prev[i]) and (close[i] < ema_4h_aligned[i]) and vol_spike_1d[i]
            
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
            # Exit when price moves back below 4h EMA50 (trend reversal)
            if close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above 4h EMA50 (trend reversal)
            if close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0