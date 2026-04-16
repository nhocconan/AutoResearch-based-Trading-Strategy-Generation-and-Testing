#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 12h EMA(34) is rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND 12h EMA(34) is falling AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, 12h EMA ensures higher timeframe trend alignment,
# volume spike confirms institutional participation. Designed to catch strong trends in both bull and bear markets.
# Target: 100-200 trades over 4 years (25-50/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA(34) for trend filter ===
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Calculate EMA slope (rising/falling)
    ema_slope = np.diff(ema_12h, prepend=ema_12h[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Align 12h EMA and slope to 4h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA, 20 for Donchian/volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        ema_val = ema_aligned[i]
        ema_rise = ema_rising_aligned[i] > 0.5
        ema_fall = ema_falling_aligned[i] > 0.5
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower band or EMA turns flat/falling
            if price < lower_band or not ema_rise:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band or EMA turns flat/rising
            if price > upper_band or not ema_fall:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band AND 12h EMA rising AND volume spike
            if price > upper_band and ema_rise and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower band AND 12h EMA falling AND volume spike
            elif price < lower_band and ema_fall and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0