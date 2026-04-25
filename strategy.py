#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Adx_Volume
Hypothesis: On 6h timeframe, use Ichimoku cloud from 1d for trend direction (price above/below cloud), 
ADX(14) from 12h for trend strength (>25), and volume confirmation on 6h (>1.5x 20-period average). 
Enter long when price above cloud, ADX>25, and volume spike; short when price below cloud, ADX>25, and volume spike.
Exit when price crosses Tenkan-Kijun (TK) cross in opposite direction or ADX drops below 20.
Designed for medium trade frequency (~20-40/year) with strong trend filtering to work in both bull and bear markets.
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
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku calculations (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX calculation (Wilder's smoothing)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # first value is simple average
            result[period-1] = np.nanmean(data[:period])
            # subsequent values: prev * (1 - 1/period) + current * (1/period)
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
                else:
                    result[i] = np.nan
            return result
        
        tr_smoothed = WilderSmoothing(tr, period)
        plus_dm_smoothed = WilderSmoothing(plus_dm, period)
        minus_dm_smoothed = WilderSmoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = WilderSmoothing(dx, period)
        
        return adx
    
    adx_values = calculate_adx(high_12h, low_12h, close_12h, period=14)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK Cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        adx_value = adx_aligned[i]
        vol_confirm = vol_spike[i]
        
        if position == 0:
            # Regime-based entry logic using Ichimoku cloud
            if close[i] > upper_cloud:  # Price above cloud = bullish bias
                # Long: price above cloud, ADX>25 (strong trend), volume spike
                long_signal = tk_bullish and (adx_value > 25) and vol_confirm
            elif close[i] < lower_cloud:  # Price below cloud = bearish bias
                # Short: price below cloud, ADX>25 (strong trend), volume spike
                short_signal = tk_bearish and (adx_value > 25) and vol_confirm
            else:
                # Price inside cloud - no clear trend, stay flat
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: 
            # 1. Price crosses below cloud (trend change)
            # 2. TK cross turns bearish (Tenkan < Kijun)
            # 3. ADX drops below 20 (trend weakening)
            exit_signal = (close[i] < upper_cloud) or (not tk_bullish) or (adx_value < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: 
            # 1. Price crosses above cloud (trend change)
            # 2. TK cross turns bullish (Tenkan > Kijun)
            # 3. ADX drops below 20 (trend weakening)
            exit_signal = (close[i] > lower_cloud) or (not tk_bearish) or (adx_value < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_Adx_Volume"
timeframe = "6h"
leverage = 1.0