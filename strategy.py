#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Volume Confirmation + ADX Trend Filter
# Williams Alligator uses smoothed moving averages (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In trending markets, the lines are ordered (Lips > Teeth > Jaw for uptrend, reverse for downtrend).
# Combined with volume confirmation to avoid false breakouts and ADX > 25 to ensure strong trends.
# Works in both bull and bear by only taking trades in the direction of the Alligator alignment.
# Target: 50-120 total trades over 4 years (12-30/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for Williams Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (13,8,5) - Smoothed Moving Averages
    def smoothed_ma(series, period):
        # Smoothed MA: first value is SMA, then smoothed
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        smoothed = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            smoothed[period-1] = sma[period-1]
            for i in range(period, len(series)):
                smoothed[i] = (smoothed[i-1] * (period-1) + series[i]) / period
        return smoothed
    
    jaw = smoothed_ma(close_1d, 13)  # Jaw (13-period)
    teeth = smoothed_ma(close_1d, 8)  # Teeth (8-period)
    lips = smoothed_ma(close_1d, 5)   # Lips (5-period)
    
    # ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM
        atr = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(tr, np.nan)
        minus_dm_smooth = np.full_like(tr, np.nan)
        
        if len(tr) >= period:
            # Initial values
            atr[period-1] = np.nanmean(tr[:period])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[:period])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[:period])
            
            # Wilder's smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        dx = np.full_like(tr, np.nan)
        
        for i in range(len(tr)):
            if not np.isnan(atr[i]) and atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX: smoothed DX
        adx = np.full_like(tr, np.nan)
        valid_dx = ~np.isnan(dx)
        if np.sum(valid_dx) >= period:
            # First ADX value is average of first 'period' DX values
            first_valid_idx = np.where(valid_dx)[0][0]
            if first_valid_idx + period <= len(tr):
                adx[first_valid_idx + period - 1] = np.nanmean(dx[first_valid_idx:first_valid_idx + period])
                for i in range(first_valid_idx + period, len(tr)):
                    if not np.isnan(dx[i]):
                        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Alligator alignment checks
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        # Long entry: Alligator aligned up (Lips > Teeth > Jaw) + volume spike + ADX > 25
        if (lips_above_teeth and teeth_above_jaw and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Alligator aligned down (Lips < Teeth < Jaw) + volume spike + ADX > 25
        elif (lips_below_teeth and teeth_below_jaw and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator alignment changes or ADX weakens (< 20)
        elif position == 1 and (not (lips_above_teeth and teeth_above_jaw) or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not (lips_below_teeth and teeth_below_jaw) or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0