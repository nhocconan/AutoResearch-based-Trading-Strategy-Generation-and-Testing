#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(50) trend filter and volume confirmation.
# Long when price breaks above 20-day high AND weekly EMA(50) is rising AND volume > 1.5x 20-day average.
# Short when price breaks below 20-day low AND weekly EMA(50) is falling AND volume > 1.5x 20-day average.
# Uses discrete position size 0.25. Donchian captures breakouts, weekly EMA ensures alignment with higher timeframe trend,
# volume spike confirms institutional participation. Designed to catch strong trending moves while avoiding chop.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Indicators: Donchian(20) ===
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_ma.values
    donchian_low = low_ma.values
    
    # === Daily Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma.values)
    
    # Get weekly data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: EMA(50) for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w_values = ema_50_1w.values
    
    # EMA slope: rising if current > previous, falling if current < previous
    ema_slope = np.diff(ema_50_1w_values, prepend=ema_50_1w_values[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Align weekly EMA slope to daily timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma.iloc[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_break = price > donchian_high[i]
        lower_break = price < donchian_low[i]
        vol_spike = volume_spike[i]
        ema_rising_val = bool(ema_rising_aligned[i])
        ema_falling_val = bool(ema_falling_aligned[i])
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below 20-day low or weekly EMA starts falling
            if lower_break or ema_falling_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above 20-day high or weekly EMA starts rising
            if upper_break or ema_rising_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 20-day high AND weekly EMA rising AND volume spike
            if upper_break and ema_rising_val and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below 20-day low AND weekly EMA falling AND volume spike
            elif lower_break and ema_falling_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0