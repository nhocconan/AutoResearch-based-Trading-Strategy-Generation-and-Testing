#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 12h HMA(21) uptrend (price > HMA) AND 4h volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 12h HMA(21) downtrend (price < HMA) AND 4h volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian provides structure, 12h HMA ensures alignment with higher timeframe trend,
# volume spike confirms participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 75-200 trades over 4 years (19-50/year) to balance opportunity and fee drag.

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
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need enough for HMA calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: HMA(21) for trend filter ===
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    
    wma_half = pd.Series(close_12h).rolling(window=half_n, min_periods=half_n).apply(
        lambda x: np.average(x, weights=np.arange(1, half_n + 1)), raw=True
    ).values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, 22)), raw=True
    ).values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).apply(
        lambda x: np.average(x, weights=np.arange(1, sqrt_n + 1)), raw=True
    ).values
    
    # Align 12h HMA21 to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian/volume, 21+sqrt(21) for HMA)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_donchian = highest_high[i]
        lower_donchian = lowest_low[i]
        hma_12h = hma_21_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian low or volume spike ends
            if price < lower_donchian or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian high or volume spike ends
            if price > upper_donchian or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > 12h HMA (uptrend) AND volume spike
            if price > upper_donchian and price > hma_12h and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND price < 12h HMA (downtrend) AND volume spike
            elif price < lower_donchian and price < hma_12h and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hHMA21_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0