#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Mean Reversion with 1d ADX regime filter and volume confirmation.
Long when Williams %R(14) < -80 (oversold) AND 1d ADX < 25 (range/weak trend) AND volume > 1.5x average.
Short when Williams %R(14) > -20 (overbought) AND 1d ADX < 25 (range/weak trend) AND volume > 1.5x average.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Designed to capture mean reversion in ranging markets while avoiding strong trends where mean reversion fails.
Works in both bull and bear markets by focusing on range-bound conditions (ADX < 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        def smooth(x, period):
            result = np.full_like(x, np.nan)
            if len(x) >= period:
                # Initial value
                result[period-1] = np.nansum(x[1:period+1])
                # Wilder smoothing
                for i in range(period, len(x)):
                    if not np.isnan(result[i-1]):
                        result[i] = result[i-1] - (result[i-1] / period) + x[i]
            return result
        
        atr = smooth(tr, period)
        dm_plus_smooth = smooth(dm_plus, period)
        dm_minus_smooth = smooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (np.abs(di_plus) + np.abs(di_minus)) * 100
        adx = smooth(dx, period)
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.values
    
    wr = williams_r(high, low, close, 14)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(wr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        wr_val = wr[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: oversold AND weak trend AND volume confirmation
            if (wr_val < -80 and adx_val < 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: overbought AND weak trend AND volume confirmation
            elif (wr_val > -20 and adx_val < 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if wr_val > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50
                if wr_val < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dADX_Volume"
timeframe = "6h"
leverage = 1.0