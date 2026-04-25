#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation. 
Long when price breaks above R1 + above 1d EMA34 + volume > 1.5x average. 
Short when price breaks below S1 + below 1d EMA34 + volume > 1.5x average.
Uses discrete sizing (0.25) to minimize fees. Target: 20-40 trades/year.
Works in bull markets via breakouts and in bear markets via trend-following shorts.
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
    
    # Get 1d data for HTF EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA34 on 1d
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume (20-period) for volume spike filter
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla pivot levels using previous day's OHLC
        # Need to get previous day's data - we'll use 1d data shifted by 1
        if i >= 16:  # Need at least 16 bars of 4h data (1 day) to calculate pivots
            # Get index of 1d data corresponding to previous day
            # Since we're on 4h timeframe, 16 bars = 1 day
            idx_1d = i // 16
            if idx_1d > 0 and idx_1d < len(df_1d):
                # Previous day's OHLC (1d data)
                prev_high = df_1d['high'].iloc[idx_1d - 1]
                prev_low = df_1d['low'].iloc[idx_1d - 1]
                prev_close = df_1d['close'].iloc[idx_1d - 1]
                
                # Calculate Camarilla levels
                range_val = prev_high - prev_low
                if range_val > 0:
                    R1 = prev_close + (range_val * 1.1 / 12)
                    S1 = prev_close - (range_val * 1.1 / 12)
                    
                    # Volume spike condition
                    volume_spike = volume[i] > (1.5 * avg_volume[i])
                    
                    if position == 0:
                        # Long: price breaks above R1 + above 1d EMA34 + volume spike
                        long_signal = (close[i] > R1) and (close[i] > ema_34_aligned[i]) and volume_spike
                        # Short: price breaks below S1 + below 1d EMA34 + volume spike
                        short_signal = (close[i] < S1) and (close[i] < ema_34_aligned[i]) and volume_spike
                        
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
                        # Exit when price closes below S1 (trend weakness) or below EMA34
                        exit_signal = (close[i] < S1) or (close[i] < ema_34_aligned[i])
                        if exit_signal:
                            signals[i] = 0.0
                            position = 0
                    elif position == -1:
                        # Short: hold position
                        signals[i] = -0.25
                        # Exit when price closes above R1 (trend reversal) or above EMA34
                        exit_signal = (close[i] > R1) or (close[i] > ema_34_aligned[i])
                        if exit_signal:
                            signals[i] = 0.0
                            position = 0
                else:
                    signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
        else:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0