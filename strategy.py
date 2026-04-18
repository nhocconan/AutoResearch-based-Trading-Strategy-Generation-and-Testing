#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout + 1d EMA trend + volume confirmation.
# Camarilla pivot levels from daily data provide key support/resistance levels.
# Breakout above R1 or below S1 with volume confirmation captures momentum.
# 1d EMA50 filters trend direction to avoid counter-trend trades.
# Designed to work in both bull and bear markets by following momentum with trend filter.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_Camarilla_R1_S1_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels on 1d data
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # Previous day's values for Camarilla calculation
    prev_high = high_1d.shift(1)
    prev_low = low_1d.shift(1)
    prev_close = close_1d.shift(1)
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_r1 = prev_close + (range_ * 1.1 / 12)
    camarilla_s1 = prev_close - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (wait for previous day's close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Get 1d data for EMA trend filter
    # Calculate EMA50 on 1d data
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Close above R1 AND price above EMA50 AND volume spike
            if close_val > r1_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 AND price below EMA50 AND volume spike
            elif close_val < s1_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below EMA50 (trend change)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above EMA50 (trend change)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals