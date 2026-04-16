#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA(50) trend filter and 1d volume confirmation.
# Long when price breaks above Donchian upper AND 1w close > EMA50 (bullish regime) AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower AND 1w close < EMA50 (bearish regime) AND 1d volume > 1.5x 20-period average.
# Exit when price crosses Donchian middle (20-period average of high/low) OR 1d volume drops below average.
# Uses discrete position size 0.25. Designed to capture strong trends in both bull and bear markets using weekly trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20) ===
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper = high_ma.values
    lower = low_ma.values
    middle = ((upper + lower) / 2)
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) for trend filter ===
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for volume MA
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align 1d volume spike to 6h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 55 periods needed for 1w EMA, 20 for Donchian/volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema_50_val = ema_50_aligned[i]
        vol_spike_1d = bool(volume_spike_1d_aligned[i])
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian middle OR 1d volume spike ends
            if price < middle[i] or not vol_spike_1d:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian middle OR 1d volume spike ends
            if price > middle[i] or not vol_spike_1d:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND 1w close > EMA50 (bullish) AND 1d volume spike
            if price > upper[i] and close[-1] > ema_50_val and vol_spike_1d:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND 1w close < EMA50 (bearish) AND 1d volume spike
            elif price < lower[i] and close[-1] < ema_50_val and vol_spike_1d:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1wEMA50_1dVolumeSpike_V1"
timeframe = "6h"
leverage = 1.0