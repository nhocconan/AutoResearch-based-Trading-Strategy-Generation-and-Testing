#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_HTFVolume
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts from the previous 12h bar with 1-week EMA50 trend filter and 1-week volume confirmation (>1.5x 20-bar average) captures institutional breakouts with controlled trade frequency. The 12h timeframe targets 12-37 trades/year (50-150 over 4 years), minimizing fee drag while allowing for significant moves. Trend alignment ensures directional bias in both bull and bear markets, volume confirms participation, and discrete sizing (0.25) minimizes fee churn. Works in bull markets via long breakouts and bear markets via short breakouts.
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
    
    # Get 1w data for HTF trend and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (20-period) on 1w for volume confirmation
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Get 12h data for Camarilla levels (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from previous 12h bar (R1, S1)
    # Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    # Use previous completed 12h bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_12h[:-1]])
    prev_high = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low = np.concatenate([[np.nan], low_12h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume average (20-period) for 12h volume spike filter
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_12h[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        vol_ma_1w_val = vol_ma_1w_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_12h_val = vol_ma_12h[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike conditions: 
        # 1. 12h volume > 1.5x 20-period average (LTF confirmation)
        # 2. 1w volume > 1.5x 20-period average (HTF participation)
        volume_spike_12h = vol_val > 1.5 * vol_ma_12h_val
        volume_spike_1w = volume_1w[min(i // (12*4), len(volume_1w)-1)] > 1.5 * vol_ma_1w_val if i >= 12*4*20 else False
        
        if position == 0:
            # Look for entry signals: Camarilla R1/S1 breakout with trend and volume
            # Long: price breaks above R1 with uptrend (close > EMA50) and volume spike
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike_12h and volume_spike_1w
            # Short: price breaks below S1 with downtrend (close < EMA50) and volume spike
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike_12h and volume_spike_1w
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S1 (exit long)
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close below EMA50 (defensive exit)
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close above EMA50 (defensive exit)
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_HTFVolume"
timeframe = "12h"
leverage = 1.0