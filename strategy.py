#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d HMA(21) is rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d HMA(21) is falling AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian breakout captures momentum, 1d HMA ensures higher timeframe trend alignment (avoiding counter-trend trades),
# volume spike confirms institutional participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 100-180 trades over 4 years (25-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for HMA calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: HMA(21) for trend filter ===
    def calculate_hma(arr, period):
        """Calculate Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA of half period
        weights_half = np.arange(1, half_period + 1)
        wma_half = np.convolve(arr, weights_half, mode='valid') / weights_half.sum()
        
        # WMA of full period
        weights_full = np.arange(1, period + 1)
        wma_full = np.convolve(arr, weights_full, mode='valid') / weights_full.sum()
        
        # HMA = 2*WMA(half) - WMA(full)
        hma_raw = 2 * wma_half - wma_full
        
        # Final WMA of sqrt period
        weights_sqrt = np.arange(1, sqrt_period + 1)
        hma = np.convolve(hma_raw, weights_sqrt, mode='valid') / weights_sqrt.sum()
        
        # Pad with NaN to match original length
        hma_padded = np.full(len(arr), np.nan)
        hma_padded[period-1:len(hma)+period-1] = hma
        
        return hma_padded
    
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # HMA slope (rising/falling) - compare current vs previous value
    hma_slope = np.diff(hma_21_1d_aligned, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods for Donchian/volume, 21+ for HMA)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(hma_21_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_spike = volume_spike[i]
        hma_rising_val = hma_rising[i]
        hma_falling_val = hma_falling[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle of channel or volume spike ends
            mid_channel = (upper_channel + lower_channel) / 2
            if price <= mid_channel or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle of channel or volume spike ends
            mid_channel = (upper_channel + lower_channel) / 2
            if price >= mid_channel or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian channel AND 1d HMA rising AND volume spike
            if price > upper_channel and hma_rising_val and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower Donchian channel AND 1d HMA falling AND volume spike
            elif price < lower_channel and hma_falling_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dHMA21_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0