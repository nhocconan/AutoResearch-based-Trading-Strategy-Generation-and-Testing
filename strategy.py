#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation.
# Long when price breaks above Donchian upper AND 1d EMA34 slope > 0 (uptrend) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower AND 1d EMA34 slope < 0 (downtrend) AND volume > 1.5x 20-period average.
# Exit when price crosses Donchian middle (20-period average of high/low).
# Uses discrete position size 0.25. Designed to capture strong trends with trend filter reducing whipsaw.
# Target: 75-200 trades over 4 years (19-50/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper = high_ma.values
    lower = low_ma.values
    middle = ((upper + lower) / 2)
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA(34) for trend filter ===
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Slope of EMA: positive = uptrend, negative = downtrend
    ema_slope = np.diff(ema_34, prepend=ema_34[0])
    
    # Align 1d EMA slope to 4h timeframe
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA, 20 for Donchian/volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema_slope_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema_slope_val = ema_slope_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian middle
            if price < middle[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian middle
            if price > middle[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND 1d EMA34 slope > 0 (uptrend) AND volume spike
            if price > upper[i] and ema_slope_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND 1d EMA34 slope < 0 (downtrend) AND volume spike
            elif price < lower[i] and ema_slope_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dEMA34Slope_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0