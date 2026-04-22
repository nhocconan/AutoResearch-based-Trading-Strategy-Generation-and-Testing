#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with weekly Donchian breakout (50-period) + volume spike + 1-day ATR volatility filter.
# Uses weekly Donchian channels for trend context (robust to gaps) and 1-day ATR for volatility.
# Designed to capture strong momentum moves in both bull and bear markets while avoiding choppy conditions.
# Targets 12-37 trades/year with disciplined risk control and discrete position sizing.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (50-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_length = 50
    
    # Upper band: highest high
    upper = pd.Series(high_1w).rolling(window=donch_length, min_periods=donch_length).max().values
    # Lower band: lowest low
    lower = pd.Series(low_1w).rolling(window=donch_length, min_periods=donch_length).min().values
    
    # Align Donchian to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    
    # Load daily data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility environments
        vol_filter = atr > 0.01 * price  # ATR > 1% of price
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        upper_band = upper_aligned[i]
        lower_band = lower_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian upper with volume and volatility
            if price > upper_band and vol_spike and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian lower with volume and volatility
            elif price < lower_band and vol_spike and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal (opposite breakout)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on break below weekly Donchian lower
                if price < lower_band:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on break above weekly Donchian upper
                if price > upper_band:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyDonchian50_VolumeSpike_VolFilter"
timeframe = "12h"
leverage = 1.0