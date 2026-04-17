#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and ATR-based regime filter.
Long when price breaks above 20-period Donchian upper band and 1d volume > 1.3x 20-period average and 1d ATR ratio < 1.0 (normal volatility).
Short when price breaks below 20-period Donchian lower band and 1d volume > 1.3x 20-period average and 1d ATR ratio < 1.0.
Exit when price returns to the midpoint of the Donchian channel or 1d ATR ratio > 1.5 (high volatility).
Uses 1d for volume and volatility regime filters, 12h for price action and entry timing.
Designed to capture medium-term breakouts in normal volatility environments which work in both bull and bear markets.
Target: 15-30 trades/year per symbol to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume and volatility filters
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 12h
    def donchian_channels(high_arr, low_arr, period=20):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    donchian_upper_12h, donchian_lower_12h, donchian_middle_12h = donchian_channels(high_12h, low_12h, 20)
    
    # Calculate 1d volume MA20
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) and ATR ratio (current ATR / 20-period average ATR)
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_1d / atr_ma_20_1d, np.nan)
    
    # Align 12h Donchian channels to 12h timeframe (they're already on 12h)
    # But we need to align to the primary timeframe (12h) for signal generation
    # Since our primary timeframe is 12h, we can use the values directly after warmup
    # However, to be safe and follow the pattern, we'll align (though it's 1:1)
    donchian_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    donchian_middle_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle_12h)
    
    # Align 1d volume MA and ATR ratio to 12h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Also align 1d volume for volume confirmation
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_12h_aligned[i]) or 
            np.isnan(donchian_lower_12h_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        # Volatility regime filter: 1d ATR ratio < 1.0 (normal volatility)
        vol_regime = atr_ratio_1d_aligned[i] < 1.0
        
        # Breakout conditions
        breakout_up = close[i] > donchian_upper_12h_aligned[i]
        breakout_down = close[i] < donchian_lower_12h_aligned[i]
        
        # Reversion condition: return to midpoint
        revert_to_middle = abs(close[i] - donchian_middle_12h_aligned[i]) < 0.001 * donchian_middle_12h_aligned[i]
        
        # High volatility exit condition
        high_vol_exit = atr_ratio_1d_aligned[i] > 1.5
        
        if position == 0:
            # Long: breakout above upper band with volume confirmation and normal volatility
            if breakout_up and volume_confirmed and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band with volume confirmation and normal volatility
            elif breakout_down and volume_confirmed and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR high volatility environment
            if revert_to_middle or high_vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR high volatility environment
            if revert_to_middle or high_vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeVolRegime"
timeframe = "12h"
leverage = 1.0