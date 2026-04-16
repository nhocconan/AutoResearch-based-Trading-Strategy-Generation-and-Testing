#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1d Williams %R (14) with 1w EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold bounce), price > 1w EMA50 (uptrend), and 1d volume > 1.5x 20-period median volume.
# Short when Williams %R crosses below -20 (overbought rejection), price < 1w EMA50 (downtrend), and same volume condition.
# Exit when Williams %R crosses -50 (mean reversion) or opposite signal triggers.
# Uses discrete position size 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# This strategy captures mean reversion in both bull and bear markets by using 1w EMA50 as regime filter and Williams %R for timing,
# avoiding whipsaws in strong trends by requiring alignment with higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Williams %R (14) and volume median ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d volume median (20-period)
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 50)  # volume median(20), Williams %R(14), EMA50(1w)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track previous Williams %R for cross detection
    prev_williams_r = np.full(n, np.nan)
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        williams_r = williams_r_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_1d = vol_1d_aligned[i]
        ema_50_1w = ema_50_1w_aligned[i]
        price = close[i]
        
        # Store previous Williams %R for cross detection
        if i > warmup:
            prev_williams_r[i] = williams_r_aligned[i-1]
        else:
            prev_williams_r[i] = williams_r  # fallback
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when Williams %R crosses below -50 (mean reversion) or short signal triggers
            if williams_r < -50 and prev_williams_r[i] >= -50:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when Williams %R crosses above -50 (mean reversion) or long signal triggers
            if williams_r > -50 and prev_williams_r[i] <= -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 1d volume > 1.5x median volume
            volume_spike = vol_1d > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Williams %R crosses above -80 (oversold bounce), price above 1w EMA50 (uptrend), and volume spike
            if (williams_r > -80 and prev_williams_r[i] <= -80 and 
                price > ema_50_1w and volume_spike):
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Williams %R crosses below -20 (overbought rejection), price below 1w EMA50 (downtrend), and volume spike
            elif (williams_r < -20 and prev_williams_r[i] >= -20 and 
                  price < ema_50_1w and volume_spike):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_WilliamsR_14_1wEMA50_1dVolumeSpike1.5x_v1"
timeframe = "1d"
leverage = 1.0