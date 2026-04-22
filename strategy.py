#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 1D Williams %R (14) + Weekly Supertrend (10, 3) trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions on daily timeframe.
# Weekly Supertrend filters trades to only take in direction of higher timeframe trend.
# Volume confirmation ensures institutional participation.
# Works in both bull and bear markets by combining mean reversion (Williams %R) with trend filter.
# Targets 10-25 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Williams %R (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Handle division by zero
    
    # Align Williams %R to daily timeframe (no additional delay needed as it's based on completed daily bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Load weekly data for Supertrend (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR (10-period) for Supertrend
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend (10, 3)
    # Basic Upper Band = (High + Low)/2 + multiplier * ATR
    # Basic Lower Band = (High + Low)/2 - multiplier * ATR
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + 3 * atr
    lower_band = hl2 - 3 * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w)
    trend = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            trend[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend to daily timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    # Calculate 20-day average volume for volume spike detection
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(supertrend_aligned[i]) or 
            np.isnan(trend_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = df_1d['volume'].values[i] if i < len(df_1d['volume']) else 0
        vol_ma = vol_ma_aligned[i]
        wr = williams_r_aligned[i]
        st = supertrend_aligned[i]
        tr_dir = trend_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R oversold (-80 or below) + uptrend + volume spike
            if wr <= -80 and tr_dir == 1 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (-20 or above) + downtrend + volume spike
            elif wr >= -20 and tr_dir == -1 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R reaches overbought (-20 or above) or trend changes
                if wr >= -20 or tr_dir == -1:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R reaches oversold (-80 or below) or trend changes
                if wr <= -80 or tr_dir == 1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_Supertrend_Volume"
timeframe = "1d"
leverage = 1.0