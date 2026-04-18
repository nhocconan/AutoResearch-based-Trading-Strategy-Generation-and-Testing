#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime
Hypothesis: Price breaking above Camarilla R1 or below S1 on 4h with volume confirmation and ADX trend filter captures institutional breakouts. Works in bull (breakouts continue) and bear (breakdowns continue) regimes. Uses 1d Camarilla levels for structure, volume > 1.5x average for confirmation, and ADX > 25 to avoid false breakouts in ranging markets. Targets 20-30 trades/year with position size 0.25.
"""

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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day has no previous
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h (wait for day close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter (14-period)
    # Calculate +DM, -DM, TR
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = np.mean(data[max(0, i-period+1):i+1]) if i >= period-1 else np.nan
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    smoothed_plus_dm = wilders_smooth(plus_dm, period)
    smoothed_minus_dm = wilders_smooth(minus_dm, period)
    smoothed_tr = wilders_smooth(tr, period)
    
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume and ADX > 25
            if (close[i] > R1_aligned[i] and volume_confirm[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume and ADX > 25
            elif (close[i] < S1_aligned[i] and volume_confirm[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal) or ADX drops < 20
            if (close[i] < S1_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal) or ADX drops < 20
            if (close[i] > R1_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0