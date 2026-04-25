#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation. Long when price breaks above R1 + EMA50 up + volume spike, short when breaks below S1 + EMA50 down + volume spike. Designed for 12-37 trades/year on 12h timeframe, works in bull markets via breakouts with trend and in bear markets via mean reversion at extreme levels with volume exhaustion.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Use previous completed 1d bar to avoid look-ahead
    shift_high_1d = np.roll(high_1d, 1)
    shift_low_1d = np.roll(low_1d, 1)
    shift_close_1d = np.roll(close_1d, 1)
    shift_high_1d[0] = np.nan
    shift_low_1d[0] = np.nan
    shift_close_1d[0] = np.nan
    
    camarilla_r1_1d = shift_close_1d + (shift_high_1d - shift_low_1d) * 1.1 / 12
    camarilla_s1_1d = shift_close_1d - (shift_high_1d - shift_low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike (20-period volume > 1.5 * 50-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = vol_ma_20 > (vol_ma_50 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1 + EMA50 up + volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i-1] <= camarilla_r1_aligned[i-1]) and \
                         (ema_50_aligned[i] > ema_50_aligned[i-1]) and volume_spike[i]
            # Short: price breaks below S1 + EMA50 down + volume spike
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i-1] >= camarilla_s1_aligned[i-1]) and \
                          (ema_50_aligned[i] < ema_50_aligned[i-1]) and volume_spike[i]
            
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
            # Exit when price closes below S1 (mean reversion) or EMA50 turns down
            exit_signal = (close[i] < camarilla_s1_aligned[i]) or (ema_50_aligned[i] < ema_50_aligned[i-1])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above R1 (mean reversion) or EMA50 turns up
            exit_signal = (close[i] > camarilla_r1_aligned[i]) or (ema_50_aligned[i] > ema_50_aligned[i-1])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0