#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1w trend filter + volume confirmation
# - Long when Williams %R(14) < -80 (oversold) in 1w uptrend (close > EMA50) with volume spike (>1.5x 20-period avg)
# - Short when Williams %R(14) > -20 (overbought) in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Williams %R identifies exhaustion points; 1w trend ensures alignment with higher timeframe momentum
# - Volume confirmation reduces false signals during low-participation moves

name = "6h_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w Williams %R(14)
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # 1w volume confirmation: > 1.5x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (1.5 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_spike_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to neutral territory (> -50) or opposite extreme (< -20 for early exit)
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral territory (< -50) or opposite extreme (> -80 for early exit)
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with trend and volume filters
            if vol_spike_1w_aligned[i]:
                # Long signal: Williams %R oversold (< -80) in 1w uptrend
                if (williams_r_aligned[i] < -80 and 
                    close_1w[-1] > ema_50_1w_aligned[i] if len(close_1w) > 0 else False):  # Simplified trend check
                    # Use current 1w close for trend (already aligned)
                    if close_1w[-1] > ema_50_1w_aligned[i]:  # This needs fixing - use aligned close
                        pass
                    # Correct approach: use aligned 1w close
                    # We'll compute aligned 1w close separately
                    position = 1
                    signals[i] = 0.25
                # Short signal: Williams %R overbought (> -20) in 1w downtrend
                elif (williams_r_aligned[i] > -20 and 
                      close_1w[-1] < ema_50_1w_aligned[i] if len(close_1w) > 0 else False):
                    if close_1w[-1] < ema_50_1w_aligned[i]:
                        pass
                    position = -1
                    signals[i] = -0.25
    
    # Fix: need to properly align 1w close for trend comparison
    # Recompute with proper aligned close
    signals = np.zeros(n)
    position = 0
    
    # Pre-compute aligned 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_spike_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to neutral territory (> -50)
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral territory (< -50)
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with trend and volume filters
            if vol_spike_1w_aligned[i]:
                # Long signal: Williams %R oversold (< -80) in 1w uptrend
                if (williams_r_aligned[i] < -80 and 
                    close_1w_aligned[i] > ema_50_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short signal: Williams %R overbought (> -20) in 1w downtrend
                elif (williams_r_aligned[i] > -20 and 
                      close_1w_aligned[i] < ema_50_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals