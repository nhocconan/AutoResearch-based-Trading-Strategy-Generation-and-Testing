#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND weekly EMA50 is rising AND volume > 1.5x 20-day average.
# Short when price breaks below Donchian lower channel AND weekly EMA50 is falling AND volume > 1.5x 20-day average.
# Uses discrete position size 0.25. Designed to capture strong trends in both bull and bear markets with volume confirmation to avoid false breakouts.
# Target: 40-80 trades over 4 years (10-20/year) to minimize fee drag while maintaining sufficient opportunities.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Indicators: Donchian Channel (20) ===
    # Upper channel: highest high over past 20 days
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over past 20 days
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Daily Indicators: Volume Spike (volume > 1.5x 20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get weekly data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: EMA(50) for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 60 periods needed for weekly EMA50, 20 for Donchian/volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        vol_spike = volume_spike[i]
        ema_50 = ema_50_1w_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower channel OR weekly EMA50 starts falling
            if price < lower or ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper channel OR weekly EMA50 starts rising
            if price > upper or ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper channel AND weekly EMA50 rising AND volume spike
            if price > upper and ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower channel AND weekly EMA50 falling AND volume spike
            elif price < lower and ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0