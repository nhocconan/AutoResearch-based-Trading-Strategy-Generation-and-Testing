#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-day average.
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-day average.
# Exit when price crosses Donchian midpoint or trend reverses.
# Uses discrete position size 0.30. Designed for low trade frequency (~15-25/year) to minimize fee drag.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns) by aligning with weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        ema_1w = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian midpoint or weekly trend turns down
            if price < mid or price < ema_1w:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian midpoint or weekly trend turns up
            if price > mid or price > ema_1w:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND price > 1w EMA50 (uptrend) AND volume spike
            if price > upper and price > ema_1w and vol_spike:
                signals[i] = 0.30
                position = 1
            
            # SHORT: Price breaks below Donchian lower band AND price < 1w EMA50 (downtrend) AND volume spike
            elif price < lower and price < ema_1w and vol_spike:
                signals[i] = -0.30
                position = -1
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0