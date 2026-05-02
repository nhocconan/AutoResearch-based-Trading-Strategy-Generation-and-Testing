#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d ADX > 25 ensures we only trade in trending markets to avoid whipsaws in ranging conditions
# Volume spike (>2.0 x 20-period EMA) confirms reversal validity
# Works in bull markets (oversold bounces in uptrend) and bear markets (overbought rejections in downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_WilliamsR_Reversal_1dADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    atr_1d = smma(tr, 14)
    dm_plus_smooth = smma(dm_plus, 14)
    dm_minus_smooth = smma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = smma(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R calculation (14-period) on 6h data
    def williams_r(high_arr, low_arr, close_arr, period=14):
        wr = np.full_like(close_arr, np.nan, dtype=float)
        for i in range(period-1, len(close_arr)):
            highest_high = np.max(high_arr[i-period+1:i+1])
            lowest_low = np.min(low_arr[i-period+1:i+1])
            if highest_high != lowest_low:
                wr[i] = -100 * (highest_high - close_arr[i]) / (highest_high - lowest_low)
        return wr
    
    williams_r_14 = williams_r(high, low, close, 14)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r_14[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Oversold (%R < -80) in trending market with volume confirmation
            if williams_r_14[i] < -80 and trending and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (%R > -20) in trending market with volume confirmation
            elif williams_r_14[i] > -20 and trending and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price returns to neutral territory (%R > -50) OR trend weakens
            if williams_r_14[i] > -50 or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price returns to neutral territory (%R < -50) OR trend weakens
            if williams_r_14[i] < -50 or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals