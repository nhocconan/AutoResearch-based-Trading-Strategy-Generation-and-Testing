#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# ADX > 25 indicates trending market (use Elder Ray for direction)
# ADX < 20 indicates ranging market (fade extreme Bull/Bear Power)
# Volume > 1.5x 20-bar average confirms momentum
# Designed for low trade frequency (<100 total 6h trades) to minimize fee drag while capturing
# both trend continuation and mean reversion opportunities in bull and bear markets.

name = "6h_ElderRay_ADX_Regime_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX for regime filter (using Wilder's smoothing)
    if len(df_1d) < 14:
        return np.zeros(n)
    # True Range
    tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
    tr2 = np.abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = np.abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    dm_plus = np.where((pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values) > 
                       (pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values),
                       np.maximum(pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values, 0), 0)
    dm_minus = np.where((pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values) > 
                        (pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values),
                        np.maximum(pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values, 0), 0)
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_1d = wilders_smoothing(dm_plus, 14)
    dm_minus_1d = wilders_smoothing(dm_minus, 14)
    # Avoid division by zero
    dm_plus_1d = np.where(atr_1d == 0, 0, dm_plus_1d)
    dm_minus_1d = np.where(atr_1d == 0, 0, dm_minus_1d)
    di_plus_1d = 100 * dm_plus_1d / atr_1d
    di_minus_1d = 100 * dm_minus_1d / atr_1d
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    dx_1d = np.where((di_plus_1d + di_minus_1d) == 0, 0, dx_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray components
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 14), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_1d_aligned[i] > 25:
                # TRENDING: Use Elder Ray for direction
                if bull_power[i] > 0 and volume[i] > 1.5 * avg_volume[i]:
                    signals[i] = 0.25  # Long on bull power
                    position = 1
                elif bear_power[i] < 0 and volume[i] > 1.5 * avg_volume[i]:
                    signals[i] = -0.25  # Short on bear power
                    position = -1
            else:
                # RANGING: Fade extreme Elder Ray (mean reversion)
                if bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 80) and volume[i] > 1.5 * avg_volume[i]:
                    signals[i] = -0.25  # Short on overbought
                    position = -1
                elif bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 20) and volume[i] > 1.5 * avg_volume[i]:
                    signals[i] = 0.25  # Long on oversold
                    position = 1
        elif position == 1:
            # EXIT LONG: Close if power deteriorates or volume drops
            if bull_power[i] < 0 or volume[i] < 0.5 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # EXIT SHORT: Close if power deteriorates or volume drops
            if bear_power[i] > 0 or volume[i] < 0.5 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals