#!/usr/bin/env python3
# 4h_donchian_volume_chop_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and chop regime filter (CHOP(14) > 61.8 for mean reversion, < 38.2 for trend following). 
# In choppy markets (CHOP > 61.8): fade breaks (short upper band, long lower band). 
# In trending markets (CHOP < 38.2): follow breaks (long upper band, short lower band). 
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 25-40 trades/year.
# Uses 1d HTF data for ADX regime confirmation (ADX > 25 = trending) as secondary filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_v3"
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
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Calculate ADX on daily timeframe
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_d[1:] - high_d[:-1]) > (low_d[:-1] - low_d[1:]), 
                       np.maximum(high_d[1:] - high_d[:-1], 0), 0)
    dm_minus = np.where((low_d[:-1] - low_d[1:]) > (high_d[1:] - high_d[:-1]), 
                        np.maximum(low_d[:-1] - low_d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, period)
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4h indicators
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period)
    def choppiness_index(high, low, close, period):
        atr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                atr_sum[i] = np.nan
            else:
                tr_sum = 0
                for j in range(i-period+1, i+1):
                    tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                    tr_sum += tr
                atr_sum[i] = tr_sum
        
        # Calculate CHOP
        chop = np.full_like(close, np.nan)
        max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        for i in range(len(close)):
            if i < period or np.isnan(atr_sum[i]) or atr_sum[i] == 0:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filters
        is_choppy = chop[i] > 61.8  # Mean reversion regime
        is_trending = chop[i] < 38.2  # Trend following regime
        strong_trend = adx_aligned[i] > 25  # Additional trend strength filter
        
        if position == 1:  # Long position
            # Exit conditions
            if is_choppy:
                # In choppy regime: exit when price reaches midpoint (mean reversion target)
                midpoint = (donch_high[i] + donch_low[i]) / 2
                if close[i] >= midpoint:
                    position = 0
                    signals[i] = 0.0
            else:
                # In trending regime: exit when price breaks lower band OR ADX weakens
                if close[i] < donch_low[i] or adx_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if is_choppy:
                # In choppy regime: exit when price reaches midpoint
                midpoint = (donch_high[i] + donch_low[i]) / 2
                if close[i] <= midpoint:
                    position = 0
                    signals[i] = 0.0
            else:
                # In trending regime: exit when price breaks upper band OR ADX weakens
                if close[i] > donch_high[i] or adx_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                if is_choppy:
                    # In choppy regime: fade the breakout
                    if close[i] > donch_high[i]:  # Price above upper band -> short
                        position = -1
                        signals[i] = -0.25
                    elif close[i] < donch_low[i]:  # Price below lower band -> long
                        position = 1
                        signals[i] = 0.25
                elif is_trending and strong_trend:
                    # In trending regime with strong ADX: follow the breakout
                    if close[i] > donch_high[i]:  # Price above upper band -> long
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < donch_low[i]:  # Price below lower band -> short
                        position = -1
                        signals[i] = -0.25
    
    return signals