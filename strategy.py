#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h for optimal trade frequency (target: 20-50 trades/year)
- Entry: Price breaks above R1 (long) or below S1 (short) from previous day's Camarilla levels
- Trend filter: 12h EMA50 slope (must be rising for longs, falling for shorts)
- Volume confirmation: Current volume > 1.5x 20-period average volume
- Exit: Price closes back inside the Camarilla H3/L3 range (mean reversion zone)
- Position sizing: 0.25 (discrete level to minimize fee churn)
- Works in bull/bear markets by requiring trend alignment and volume confirmation
- Low trade frequency avoids fee drag while capturing strong breakout moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We focus on R1, S1, H3, L3 for breakout and exit
    range_1d = df_1d['high'].values - df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1 = close_1d + 1.1 * range_1d / 12
    S1 = close_1d - 1.1 * range_1d / 12
    H3 = close_1d + 1.1 * range_1d / 6
    L3 = close_1d - 1.1 * range_1d / 6
    
    # Align Camarilla levels to 4h timeframe (completed previous day only)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate EMA50 slope (trend direction)
    ema50_slope = np.zeros_like(ema50_12h_aligned)
    ema50_slope[1:] = ema50_12h_aligned[1:] - ema50_12h_aligned[:-1]
    # Align slope to 4h timeframe
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 50 for EMA50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > R1_aligned[i] and volume_spike[i]
        breakout_short = close[i] < S1_aligned[i] and volume_spike[i]
        
        # Trend filter conditions
        uptrend = ema50_slope_aligned[i] > 0  # Rising EMA50
        downtrend = ema50_slope_aligned[i] < 0  # Falling EMA50
        
        # Exit conditions: price returns to mean reversion zone (H3/L3)
        exit_long = close[i] < H3_aligned[i]
        exit_short = close[i] > L3_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + uptrend
            if breakout_long and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume spike + downtrend
            elif breakout_short and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price returns to H3 (mean reversion) OR trend changes
            if exit_long or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price returns to L3 (mean reversion) OR trend changes
            if exit_short or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0