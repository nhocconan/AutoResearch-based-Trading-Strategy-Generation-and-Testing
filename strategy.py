#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d ADX Trend + Volume Spike
# Long when Williams %R crosses above -80 from below, ADX > 25, and 1d volume > 1.5x 20-day average
# Short when Williams %R crosses below -20 from above, ADX > 25, and 1d volume > 1.5x 20-day average
# Exit when Williams %R crosses -50 (middle) or ADX drops below 20
# Williams %R identifies overbought/oversold conditions, ADX filters for trending markets
# Volume confirms conviction, works in both bull (oversold bounces) and bear (overbought reversals)
# Target: 20-35 trades/year by requiring ADX trend filter + volume spike

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-period Williams %R
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Calculate 14-period ADX
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = None
    
    for i in range(14, n):  # Start after Williams %R/ADX warmup
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            prev_williams_r = williams_r_aligned[i] if not np.isnan(williams_r_aligned[i]) else prev_williams_r
            continue
        
        # Current values
        williams = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Get current 1d volume (12 bars per day for 12h timeframe)
        vol_idx = i // 12
        if vol_idx >= len(df_1d):
            vol_idx = len(df_1d) - 1
        volume = df_1d['volume'].iloc[vol_idx] if vol_idx >= 0 else df_1d['volume'].iloc[0]
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below, ADX > 25, volume confirmation
            if (prev_williams_r is not None and 
                prev_williams_r <= -80 and williams > -80 and 
                adx_val > 25 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, ADX > 25, volume confirmation
            elif (prev_williams_r is not None and 
                  prev_williams_r >= -20 and williams < -20 and 
                  adx_val > 25 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R crosses above -50 or ADX drops below 20
                if (prev_williams_r is not None and 
                    prev_williams_r < -50 and williams >= -50) or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R crosses below -50 or ADX drops below 20
                if (prev_williams_r is not None and 
                    prev_williams_r > -50 and williams <= -50) or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
        
        prev_williams_r = williams
    
    return signals

name = "12h_WilliamsR_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0