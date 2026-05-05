#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes for mean reversion in ranging markets, filtered by 1w ADX trend strength
# Long when 1d Williams %R < -80 (oversold) AND 1w ADX < 25 (ranging/weak trend) AND price > 6h VWAP
# Short when 1d Williams %R > -20 (overbought) AND 1w ADX < 25 AND price < 6h VWAP
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR ADX > 30 (trend emerging)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 60-120 total trades over 4 years (15-30/year) for 6h timeframe
# Williams %R identifies overextended moves in ranging markets
# 1w ADX filter ensures we only trade when higher timeframe lacks strong trend (avoids fighting trends)
# 6h VWAP provides dynamic support/resistance for entry timing
# Works in bull markets (mean reversion during rallies) and bear markets (mean reversion during declines)

name = "6h_WilliamsR_MeanReversion_1wADXFilter_VWAP"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least one completed 1d bar for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_1d[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for 1w ADX calculation
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX (Average Directional Index)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original arrays
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[1:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = WilderSmoothing(tr, period)
        dm_plus_smooth = WilderSmoothing(dm_plus, period)
        dm_minus_smooth = WilderSmoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # Directional Index
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        # ADX: smoothed DX
        adx = WilderSmoothing(dx, period)
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, period=14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 6h VWAP (Volume Weighted Average Price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vwap[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), weak trend (ADX < 25), price above VWAP
            if williams_r_aligned[i] < -80 and adx_1w_aligned[i] < 25 and close[i] > vwap[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), weak trend (ADX < 25), price below VWAP
            elif williams_r_aligned[i] > -20 and adx_1w_aligned[i] < 25 and close[i] < vwap[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR ADX > 30 (trend emerging)
            if williams_r_aligned[i] > -50 or adx_1w_aligned[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR ADX > 30 (trend emerging)
            if williams_r_aligned[i] < -50 or adx_1w_aligned[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals