#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot levels (from 1d) + 1d volume spike + 1d chop regime filter
    # Long: price touches Camarilla L3 support AND 1d volume > 1.8 * 20-period average AND chop > 61.8 (range)
    # Short: price touches Camarilla H3 resistance AND 1d volume > 1.8 * 20-period average AND chop > 61.8 (range)
    # Exit: price reverts to Camarilla pivot point (mid) OR chop < 38.2 (trending)
    # Using 1d for Camarilla calculation, volume, and chop to avoid look-ahead
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 25-50 trades/year (~100-200 over 4 years) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivots, volume, and chop (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # MPP = (high + low + close) / 3
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Calculate Camarilla levels
    camarilla_mpp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align 1d Camarilla to 4h (wait for completed 1d bar)
    camarilla_mpp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mpp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume spike filter: volume > 1.8 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * vol_ma_20)
    
    # Calculate 1d Choppiness Index (CHOP) - range/trend regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low)))) over period
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) - using Wilder's smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Choppiness Index (14-period)
    chop_period = 14
    sum_atr = np.full_like(atr_1d, np.nan)
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    
    for i in range(len(atr_1d)):
        if i < chop_period - 1:
            continue
        if np.isnan(atr_1d[i-chop_period+1:i+1]).any():
            continue
        sum_atr[i] = np.nansum(atr_1d[i-chop_period+1:i+1])
        highest_high[i] = np.nanmax(high_1d[i-chop_period+1:i+1])
        lowest_low[i] = np.nanmin(low_1d[i-chop_period+1:i+1])
    
    # Avoid division by zero
    range_1d = highest_high - lowest_low
    chop = np.full_like(atr_1d, 50.0)  # default to neutral
    mask = (range_1d > 0) & ~np.isnan(sum_atr)
    chop[mask] = 100 * np.log10(sum_atr[mask] / (np.log10(chop_period) * range_1d[mask]))
    
    # Align 1d indicators to 4h (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_mpp_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when chop > 61.8 (range-bound market)
        in_range = chop_aligned[i] > 61.8
        # Exit regime: chop < 38.2 (trending market) - exit positions
        in_trend = chop_aligned[i] < 38.2
        
        # Volume confirmation: 1d volume spike
        vol_confirmed = volume_spike_aligned[i] > 0.5  # boolean as float
        
        # Entry conditions: price touches Camarilla levels with volume and range regime
        # Allow small tolerance for touching levels (0.1% of price)
        tolerance = 0.001 * close[i]
        long_entry = (close[i] <= camarilla_l3_aligned[i] + tolerance) and vol_confirmed and in_range
        short_entry = (close[i] >= camarilla_h3_aligned[i] - tolerance) and vol_confirmed and in_range
        
        # Exit logic: price reverts to Camarilla pivot point OR regime shifts to trending
        long_exit = (close[i] > camarilla_mpp_aligned[i]) or in_trend
        short_exit = (close[i] < camarilla_mpp_aligned[i]) or in_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0