#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX25 regime filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX<25), 
# extreme readings (>80 for oversold, <20 for overbought) with volume spike 
# indicate high-probability reversals. In trending markets (ADX>25), we avoid counter-trend trades.
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing mean reversion edges.

name = "6h_WilliamsR_Extreme_1dADX25_VolumeSpike_Regime"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr > 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe (use completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    highest_high = np.maximum.accumulate(high_6h)
    lowest_low = np.minimum.accumulate(low_6h)
    
    # For rolling window, we need to look back 14 periods
    williams_r = np.full_like(close_6h, np.nan)
    for i in range(13, len(close_6h)):
        period_high = np.max(high_6h[i-13:i+1])
        period_low = np.min(low_6h[i-13:i+1])
        if period_high != period_low:
            williams_r[i] = (period_high - close_6h[i]) / (period_high - period_low) * -100
        else:
            williams_r[i] = -50  # Neutral when no range
    
    # Align Williams %R to 6h timeframe (already on 6h, but align for safety)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 6h data for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade mean reversion in ranging markets (ADX < 25)
        is_ranging = adx_aligned[i] < 25
        
        # Williams %R extreme conditions
        williams_oversold = williams_r_aligned[i] < -80  # Oversold
        williams_overbought = williams_r_aligned[i] > -20  # Overbought
        
        if position == 0:
            # Long: Williams %R oversold in ranging market with volume spike
            if williams_oversold and is_ranging and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought in ranging market with volume spike
            elif williams_overbought and is_ranging and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 (neutral) or regime changes to trending
            if williams_r_aligned[i] > -50 or not is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 (neutral) or regime changes to trending
            if williams_r_aligned[i] < -50 or not is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals