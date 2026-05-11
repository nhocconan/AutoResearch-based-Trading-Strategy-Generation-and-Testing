#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation, trend filter, and volume
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d close for Camarilla pivot levels
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d average volume for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous 1d bar (i-16 because 16*4h=1d)
        # But we must use completed 1d bar only - align_htf_to_ltf handles this
        # For Camarilla, we need the previous completed 1d bar's OHLC
        # Since we're on 4h timeframe, we look back 16 bars for the previous day
        if i >= 16:
            # Get the 1d bar that closed 16*4h=64 hours ago (completed)
            idx_1d = (i // 16) - 1  # Previous completed 1d bar
            if idx_1d >= 0 and idx_1d < len(df_1d):
                # Previous day's OHLC
                high_1d = df_1d['high'].iloc[idx_1d]
                low_1d = df_1d['low'].iloc[idx_1d]
                close_1d_prev = df_1d['close'].iloc[idx_1d]
                
                # Calculate Camarilla levels
                range_1d = high_1d - low_1d
                if range_1d > 0:
                    # Resistance levels
                    R3 = close_1d_prev + (range_1d * 1.1 / 6)
                    R4 = close_1d_prev + (range_1d * 1.1 / 2)
                    # Support levels
                    S3 = close_1d_prev - (range_1d * 1.1 / 6)
                    S4 = close_1d_prev - (range_1d * 1.1 / 2)
                    
                    # Volume spike: current volume > 2x 20-period EMA
                    volume_spike = volume[i] > (vol_ma_1d_aligned[i] * 2)
                    
                    if position == 0:
                        # Long: price breaks above R3 AND above 1d EMA34 (uptrend) AND volume spike
                        if close[i] > R3 and close[i] > ema_1d_aligned[i] and volume_spike:
                            signals[i] = 0.25
                            position = 1
                        # Short: price breaks below S3 AND below 1d EMA34 (downtrend) AND volume spike
                        elif close[i] < S3 and close[i] < ema_1d_aligned[i] and volume_spike:
                            signals[i] = -0.25
                            position = -1
                    elif position == 1:
                        # Long exit: price falls below S3 OR below 1d EMA34
                        if close[i] < S3 or close[i] < ema_1d_aligned[i]:
                            signals[i] = 0.0
                            position = 0
                        else:
                            signals[i] = 0.25  # maintain position
                    elif position == -1:
                        # Short exit: price rises above R3 OR above 1d EMA34
                        if close[i] > R3 or close[i] > ema_1d_aligned[i]:
                            signals[i] = 0.0
                            position = 0
                        else:
                            signals[i] = -0.25  # maintain position
                else:
                    if position != 0:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.0
            else:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals