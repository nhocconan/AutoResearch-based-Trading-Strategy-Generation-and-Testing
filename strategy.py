#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 12h EMA50 uptrend (price > EMA50) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 12h EMA50 downtrend (price < EMA50) AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, 12h EMA ensures alignment with higher timeframe trend,
# volume spike confirms participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 100-180 trades over 4 years (25-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA50 for trend filter ===
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        ema_12h = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian low or volume spike ends
            if price < dc_low or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian high or volume spike ends
            if price > dc_high or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > 12h EMA50 (uptrend) AND volume spike
            if price > dc_high and price > ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND price < 12h EMA50 (downtrend) AND volume spike
            elif price < dc_low and price < ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0