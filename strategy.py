#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation.
# Long when close breaks above Donchian upper (20-period high) AND 1w EMA(50) rising AND volume > 1.5x 20-period average.
# Short when close breaks below Donchian lower (20-period low) AND 1w EMA(50) falling AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, 1w EMA ensures alignment with weekly trend (avoiding counter-trend trades),
# volume spike confirms institutional participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 30-100 trades over 4 years (7-25/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    # Upper band: highest high over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) for trend filter ===
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA slope: rising if current > previous, falling if current < previous
    ema_rising = np.zeros_like(ema_50, dtype=bool)
    ema_falling = np.zeros_like(ema_50, dtype=bool)
    ema_rising[1:] = ema_50[1:] > ema_50[:-1]
    ema_falling[1:] = ema_50[1:] < ema_50[:-1]
    
    # Align 1w EMA slope to 1d timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_spike = volume_spike[i]
        ema_rise = ema_rising_aligned[i]
        ema_fall = ema_falling_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price closes below Donchian lower or volume spike ends
            if price < donch_low or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price closes above Donchian upper or volume spike ends
            if price > donch_high or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND 1w EMA rising AND volume spike
            if price > donch_high and ema_rise and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND 1w EMA falling AND volume spike
            elif price < donch_low and ema_fall and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0