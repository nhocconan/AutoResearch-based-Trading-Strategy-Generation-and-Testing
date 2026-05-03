#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX < 25),
# extreme readings (%R < -80 for long, %R > -20 for short) signal mean reversion.
# Volume spike confirms conviction. Designed for 12-30 trades/year on 6h to minimize fee drag
# while working in both bull (via mean reversion in ranges) and bear (via trend filters).

name = "6h_WilliamsR_Extreme_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1d ADX(25) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value: simple average
            result[period-1] = np.nansum(data[1:period])  # skip index 0 (nan)
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = WilderSmooth(tr, 25)
    dm_plus_smooth = WilderSmooth(dm_plus, 25)
    dm_minus_smooth = WilderSmooth(dm_minus, 25)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = WilderSmooth(dx, 25)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Williams %R(14)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient warmup for Williams %R
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Williams %R using data up to current bar
        lookback = min(14, i+1)
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        # Avoid division by zero
        if highest_high == lowest_low:
            williams_r = -50  # neutral
        else:
            williams_r = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Extreme Williams %R conditions for mean reversion in ranging markets (ADX < 25)
        williams_r_oversold = williams_r < -80
        williams_r_overbought = williams_r > -20
        ranging_market = adx_1d_aligned[i] < 25
        
        if position == 0:
            # Long: oversold in ranging market with volume spike
            if williams_r_oversold and ranging_market and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: overbought in ranging market with volume spike
            elif williams_r_overbought and ranging_market and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum shift) or ADX trends up
            if williams_r > -50 or adx_1d_aligned[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum shift) or ADX trends up
            if williams_r < -50 or adx_1d_aligned[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals