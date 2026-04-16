#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation.
# Long when close > Donchian upper(20) AND 1w EMA(50) slope > 0 AND volume > 1.5x 20-day average.
# Short when close < Donchian lower(20) AND 1w EMA(50) slope < 0 AND volume > 1.5x 20-day average.
# Uses discrete position size 0.25. Donchian captures breakouts, 1w EMA slope ensures we trade with higher timeframe trend,
# volume spike confirms institutional participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns).
# Target: 40-80 trades over 4 years (10-20/year) to minimize fee drag while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channels (20) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) slope for trend filter ===
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate slope: (current - 5 periods ago) / 5 to get weekly slope
    ema_50_shifted = np.roll(ema_50, 5)
    ema_50_shifted[:5] = np.nan
    ema_slope = (ema_50 - ema_50_shifted) / 5
    
    # Align 1w EMA slope to 1d timeframe
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 55 periods needed for EMA(50)+slope, 20 for Donchian/volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_slope_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_slope_val = ema_slope_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower or EMA slope turns negative
            if price < lower or ema_slope_val < 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper or EMA slope turns positive
            if price > upper or ema_slope_val > 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Donchian upper AND 1w EMA slope > 0 (uptrend) AND volume spike
            if price > upper and ema_slope_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < Donchian lower AND 1w EMA slope < 0 (downtrend) AND volume spike
            elif price < lower and ema_slope_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0