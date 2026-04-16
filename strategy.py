#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1w EMA200 is rising AND volume > 1.5x 20-day average.
# Short when price breaks below Donchian(20) low AND 1w EMA200 is falling AND volume > 1.5x 20-day average.
# Uses discrete position size 0.25. Donchian captures breakouts, 1w EMA200 ensures we trade with higher timeframe trend (avoiding counter-trend whipsaws),
# volume spike confirms institutional participation. Designed to work in both bull (breakouts up) and bear (breakdowns down) markets.
# Target: 40-80 trades over 4 years (10-20/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA200 for trend filter ===
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # EMA200 slope (rising/falling) - using 5-period change
    ema_200_slope = np.zeros_like(ema_200_aligned)
    ema_200_slope[5:] = ema_200_aligned[5:] - ema_200_aligned[:-5]
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200, 20 for Donchian/volume MA)
    warmup = 220
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(ema_200_slope[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        ema_slope = ema_200_slope[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (failed breakout) or EMA slope turns negative
            if price < lowest_low[i] or ema_slope <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (failed breakdown) or EMA slope turns positive
            if price > highest_high[i] or ema_slope >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND EMA200 rising AND volume spike
            if price > highest_high[i] and ema_slope > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND EMA200 falling AND volume spike
            elif price < lowest_low[i] and ema_slope < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA200_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0