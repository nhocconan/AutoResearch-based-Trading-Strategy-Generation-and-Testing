#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams %R extremes + 1d EMA trend filter + volume spike confirmation.
# Long when 12h Williams %R < -80 (oversold), price > 1d EMA50, and volume > 1.5x 20-period average.
# Short when 12h Williams %R > -20 (overbought), price < 1d EMA50, and volume > 1.5x 20-period average.
# Williams %R identifies exhaustion points in ranging/bear markets, EMA50 provides trend bias,
# volume spike confirms conviction. Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R calculation (14-period)
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i < 13 or np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or highest_high_14[i] == lowest_low_14[i]:
            williams_r[i] = -50.0  # neutral
        else:
            williams_r[i] = (highest_high_14[i] - close_12h[i]) / (highest_high_14[i] - lowest_low_14[i]) * -100
    
    # Williams %R signals: 1 for oversold (< -80), -1 for overbought (> -20), 0 otherwise
    williams_signal = np.zeros_like(williams_r)
    williams_signal[williams_r < -80] = 1   # oversold -> long signal
    williams_signal[williams_r > -20] = -1  # overbought -> short signal
    
    # Align 12h Williams %R signal to 4h timeframe
    williams_signal_aligned = align_htf_to_ltf(prices, df_12h, williams_signal)
    
    # Get 1d data once before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h data for volume and price
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume moving average (20-period) on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_signal_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        williams_val = williams_signal_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R signal turns negative (overbought) or price drops below EMA50
            if williams_val <= 0 or price < ema_50_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R signal turns positive (oversold) or price rises above EMA50
            if williams_val >= 0 or price > ema_50_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: price must be on correct side of EMA50
            price_filter_long = price > ema_50_val
            price_filter_short = price < ema_50_val
            
            # Volume filter: volume > 1.5x 20-period average (4h)
            vol_filter = vol > 1.5 * vol_ma_val
            
            # LONG: Williams %R oversold, price > EMA50, volume spike
            if (williams_val > 0) and price_filter_long and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R overbought, price < EMA50, volume spike
            elif (williams_val < 0) and price_filter_short and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_12hWilliamsR_EMA50_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0