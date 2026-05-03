#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 AND Bear Power < previous Bear Power (strengthening) AND 1d ADX > 25 AND volume > 1.5x 20-period MA.
# Short when Bear Power < 0 AND Bull Power < previous Bull Power AND 1d ADX > 25 AND volume > 1.5x 20-period MA.
# Exit when power weakens (Bull Power < previous for longs, Bear Power < previous for shorts) OR ADX < 20 (regime change to ranging).
# Uses 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Elder Ray measures price strength relative to EMA, ADX filters for trending markets only, volume confirms participation.
# Designed to work in both bull (strong Bull Power) and bear (strong Bear Power) markets by trading with the trend.

name = "6h_ElderRay_1dADX_VolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX and EMA13 (for Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX (trend strength filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track previous power values for strengthening condition
    prev_bull_power = np.zeros(n)
    prev_bear_power = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            # Carry forward previous power values
            if i > 0:
                prev_bull_power[i] = prev_bull_power[i-1]
                prev_bear_power[i] = prev_bear_power[i-1]
            continue
            
        # Calculate Elder Ray components
        bull_power = high[i] - ema_13_1d_aligned[i]
        bear_power = ema_13_1d_aligned[i] - low[i]
        # Store for next iteration
        if i < n-1:
            prev_bull_power[i+1] = bull_power
            prev_bear_power[i+1] = bear_power
        
        # Volume spike condition: current 6h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # 1d ADX conditions
        adx_trending = adx_1d_aligned[i] > 25
        adx_ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long: Bull Power > 0 AND strengthening (current > previous) AND trending AND volume spike AND session
            if bull_power > 0 and bull_power > prev_bull_power[i] and adx_trending and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND strengthening (current > previous) AND trending AND volume spike AND session
            elif bear_power > 0 and bear_power > prev_bear_power[i] and adx_trending and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power weakening (current < previous) OR ADX becomes ranging
            if bull_power < prev_bull_power[i] or adx_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power weakening (current < previous) OR ADX becomes ranging
            if bear_power < prev_bear_power[i] or adx_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals