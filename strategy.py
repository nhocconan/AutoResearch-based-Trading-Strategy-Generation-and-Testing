#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1, S1) with volume confirmation and 1w ADX regime filter.
# Long when price breaks above R1 with volume > 1.5x average AND 1w ADX < 25 (ranging/low trend).
# Short when price breaks below S1 with volume > 1.5x average AND 1w ADX < 25.
# Exit when price reverts to 1d close (mean reversion) or ADX > 30 (strong trend).
# Uses discrete position size 0.25. Camarilla pivots provide intraday support/resistance levels proven effective in crypto.
# 1w ADX filter avoids trading in strong trends where mean reversion fails.
# 4h timeframe targets 20-40 trades/year to minimize fee drag while capturing meaningful moves.
# Works in ranging markets (mean reversion at pivots) and avoids strong trending markets where breakouts fail.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Classic Camarilla formula based on previous day's range
    # Pivot = (H + L + C) / 3
    # R1 = C + ((H - L) * 1.1 / 12)
    # S1 = C - ((H - L) * 1.1 / 12)
    # Using previous day's data to avoid look-ahead
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    range_1d = high_1d[:-1] - low_1d[:-1]
    r1_1d = close_1d[:-1] + (range_1d * 1.1 / 12.0)
    s1_1d = close_1d[:-1] - (range_1d * 1.1 / 12.0)
    
    # Prepend NaN for first day (no previous day data)
    pivot_1d = np.concatenate([[np.nan], pivot_1d])
    r1_1d = np.concatenate([[np.nan], r1_1d])
    s1_1d = np.concatenate([[np.nan], s1_1d])
    
    # === 1w Indicators: ADX for regime filter ===
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    period = 14
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # +DM and -DM
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(values)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = values[i] * alpha + result[i-1] * (1 - alpha)
        return result
    
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
    minus_di = 100 * (minus_dm_smoothed / tr_smoothed)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0.0)
    adx = wilders_smoothing(dx, period)
    
    # Align all indicators to primary timeframe (4h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100  # sufficient for 1d pivots + 1w ADX + volume MA
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        pivot = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_avg = vol_ma[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reverts to pivot (mean reversion) OR ADX > 30 (strong trend)
            if (price <= pivot) or (adx_val > 30.0):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reverts to pivot (mean reversion) OR ADX > 30 (strong trend)
            if (price >= pivot) or (adx_val > 30.0):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x average
            volume_confirm = vol > (1.5 * vol_avg)
            
            # Regime filter: 1w ADX < 25 (ranging/low trend environment)
            regime_filter = adx_val < 25.0
            
            # LONG: Price breaks above R1 with volume confirmation AND ranging regime
            if (price > r1) and volume_confirm and regime_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 with volume confirmation AND ranging regime
            elif (price < s1) and volume_confirm and regime_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dCamarilla_R1S1_VolumeSpike_1wADXRegimeFilter_V1"
timeframe = "4h"
leverage = 1.0