#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation
# Uses Donchian channels from 4h chart to identify breakouts in direction of 1d trend.
# Enters long when price breaks above 20-period high with volume confirmation and 1d HMA21 uptrend.
# Enters short when price breaks below 20-period low with volume confirmation and 1d HMA21 downtrend.
# Designed for 19-50 trades/year (~75-200 total over 4 years) to minimize fee drag.
# Donchian provides structure, volume confirms breakout validity, HMA21 filters trend.
# Works in bull markets via breakouts and in bear markets via breakdowns.

name = "4h_Donchian20_HMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA21 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate HMA21 on daily close
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma2 = np.convolve(arr, np.ones(half_period)/half_period, mode='same')
        wma1 = 2 * np.convolve(arr, np.ones(period)/period, mode='same')
        wma3 = np.convolve(arr, np.ones(sqrt_period)/sqrt_period, mode='same')
        wma2[:half_period-1] = np.nan
        wma2[-half_period:] = np.nan
        wma1[:period-1] = np.nan
        wma1[-period:] = np.nan
        wma3[:sqrt_period-1] = np.nan
        wma3[-sqrt_period:] = np.nan
        raw = 2 * wma2 - wma1
        hma_vals = np.convolve(raw, np.ones(sqrt_period)/sqrt_period, mode='same')
        hma_vals[:sqrt_period-1] = np.nan
        hma_vals[-sqrt_period:] = np.nan
        return hma_vals
    
    hma21_1d = hma(close_1d, 21)
    
    # Align HMA21 to 4h timeframe (wait for completed 1d bar)
    hma21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma21_1d)
    
    # Calculate Donchian channels on 4h chart
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma21_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND 1d HMA21 uptrend
            if (close[i] > donchian_upper[i] and 
                volume_spike[i] and 
                close[i] > hma21_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND 1d HMA21 downtrend
            elif (close[i] < donchian_lower[i] and 
                  volume_spike[i] and 
                  close[i] < hma21_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR trend reverses
            if (close[i] >= donchian_lower[i] and close[i] <= donchian_upper[i]) or close[i] < hma21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR trend reverses
            if (close[i] >= donchian_lower[i] and close[i] <= donchian_upper[i]) or close[i] > hma21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals