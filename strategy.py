#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d VWAP trend filter with volume spike confirmation
# Williams %R identifies overbought/oversold conditions. Readings below -80 = oversold, above -20 = overbought.
# Strategy: In ranging markets (identified by low ADX), buy when Williams %R crosses above -80 from below
#           Sell when Williams %R crosses below -20 from above
#           In trending markets (ADX > 25), only take trades in direction of 1d VWAP trend
#           Volume spike (>2x 20-period average) confirms institutional participation
#           Designed for ~25-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    diff = highest_high - lowest_low
    diff = np.where(diff == 0, 1e-10, diff)
    willr = -100 * ((highest_high - close) / diff)
    
    # ADX calculation for trend strength (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    atr[13] = np.mean(tr[0:14])
    dm_plus_smooth[13] = np.mean(dm_plus[0:14])
    dm_minus_smooth[13] = np.mean(dm_minus[0:14])
    
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Avoid division by zero
    dm_plus_di = 100 * dm_plus_smooth / np.where(atr == 0, 1e-10, atr)
    dm_minus_di = 100 * dm_minus_smooth / np.where(atr == 0, 1e-10, atr)
    dx = np.abs(dm_plus_di - dm_minus_di) / np.where((dm_plus_di + dm_minus_di) == 0, 1e-10, (dm_plus_di + dm_minus_di)) * 100
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX value after 2*period
    for i in range(28, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Get 1d data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    volume_1d = df_1d['volume'].values
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_den = np.where(vwap_den == 0, 1e-10, vwap_den)
    vwap_1d = vwap_num / vwap_den
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(adx[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime
        is_ranging = adx[i] < 25
        is_trending = adx[i] >= 25
        
        if is_ranging:
            # In ranging market, mean reversion at extremes
            if willr[i] > -80 and willr[i-1] <= -80:  # Cross above -80 (oversold exit)
                if close[i] < vwap_1d_aligned[i] and volume_filter[i]:  # Below VWAP for long bias
                    signals[i] = 0.25
                    position = 1
            elif willr[i] < -20 and willr[i-1] >= -20:  # Cross below -20 (overbought exit)
                if close[i] > vwap_1d_aligned[i] and volume_filter[i]:  # Above VWAP for short bias
                    signals[i] = -0.25
                    position = -1
        else:  # Trending market
            # Only trade in direction of 1d VWAP trend
            if close[i] > vwap_1d_aligned[i]:  # Uptrend
                if willr[i] > -80 and willr[i-1] <= -80:  # Pullback to oversold
                    if volume_filter[i]:
                        signals[i] = 0.25
                        position = 1
            elif close[i] < vwap_1d_aligned[i]:  # Downtrend
                if willr[i] < -20 and willr[i-1] >= -20:  # Pullback to overbought
                    if volume_filter[i]:
                        signals[i] = -0.25
                        position = -1
        
        # Hold position if no signal change
        if signals[i] == 0 and position != 0:
            signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_1dVWAP_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0