#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume spike confirmation.
Williams %R identifies overbought/oversold conditions; ADX > 25 filters for trending markets where mean reversion works best on pullbacks; volume spike confirms institutional participation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years). Discrete position sizing (0.25) minimizes fee churn.
Works in both bull/bear via ADX regime filter and volume confirmation to avoid false signals.
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
    volume = prices['volume'].values
    
    # Calculate 1d ADX for regime filter (trending market)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    atr = WilderSmooth(tr, period_adx)
    dm_plus_smooth = WilderSmooth(dm_plus, period_adx)
    dm_minus_smooth = WilderSmooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, period_adx)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R (14-period)
    def williams_r(high_arr, low_arr, close_arr, period):
        highest_high = np.full_like(high_arr, np.nan)
        lowest_low = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            highest_high[i] = np.max(high_arr[i-period+1:i+1])
            lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close_arr) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume spike confirmation (2.5x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need WR14, ADX, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 indicates trending market (good for mean reversion on pullbacks)
        trending_market = adx_aligned[i] > 25
        
        # Volume filter: volume spike confirms participation
        vol_spike = volume[i] > 2.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) in trending market with volume spike
            if wr[i] < -80 and trending_market and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) in trending market with volume spike
            elif wr[i] > -20 and trending_market and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to mean (-50) or opposite extreme
            exit_signal = False
            if position == 1:
                # Exit long when WR crosses above -50 (mean reversion complete)
                if wr[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when WR crosses below -50 (mean reversion complete)
                if wr[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0