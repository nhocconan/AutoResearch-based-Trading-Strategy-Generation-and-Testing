#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Williams %R (14) with 1d EMA200 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold rebound), price > 1d EMA200, and volume > 1.5x 20-period median.
# Short when Williams %R crosses below -20 (overbought rejection), price < 1d EMA200, and same volume condition.
# Exit when Williams %R crosses above -20 (for longs) or below -80 (for shorts).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Williams %R is a momentum oscillator that identifies overbought/oversold levels, effective in ranging and trending markets.
# The 1d EMA200 provides a strong regime filter to avoid counter-trend trades, while volume confirmation reduces false signals.
# This combination has shown robustness in both bull and bear markets for BTC/ETH.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Williams %R and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Williams %R (14) and volume median ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    # Calculate 4h Williams %R (14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 4h volume median (20-period)
    vol_median_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA200 for trend filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to primary timeframe (4h)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20, 200)  # Williams %R(14), volume median(20), EMA200(1d)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = williams_r_aligned[warmup-1] if warmup > 0 else -50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            prev_williams_r = williams_r_aligned[i]
            continue
        
        # Current values (aligned)
        williams = williams_r_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_4h = vol_4h_aligned[i]
        ema_200_1d = ema_200_1d_aligned[i]
        price = close[i]
        
        # Williams %R crossover signals
        williams_cross_above_80 = prev_williams_r < -80 and williams >= -80
        williams_cross_below_20 = prev_williams_r > -20 and williams <= -20
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when Williams %R crosses above -20 (overbought, take profit)
            if williams_cross_above_20:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when Williams %R crosses below -80 (oversold, take profit)
            if williams_cross_above_80:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            prev_williams_r = williams
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 4h volume > 1.5x median volume
            volume_spike = vol_4h > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Williams %R crosses above -80 (oversold rebound), price above 1d EMA200 (uptrend), volume spike
            if williams_cross_above_80 and price > ema_200_1d and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Williams %R crosses below -20 (overbought rejection), price below 1d EMA200 (downtrend), volume spike
            elif williams_cross_below_20 and price < ema_200_1d and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
        
        prev_williams_r = williams
    
    return signals

name = "4h_WilliamsR_1dEMA200_VolumeSpike1.5x_v1"
timeframe = "4h"
leverage = 1.0