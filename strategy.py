#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX regime filter and volume spike
# Uses Camarilla pivot levels from daily chart to identify key support/resistance levels.
# Enters long when price breaks above R3 with volume confirmation and 1d ADX > 25 (trending market).
# Enters short when price breaks below S3 with volume confirmation and 1d ADX > 25 (trending market).
# In ranging markets (ADX <= 25), uses mean reversion at Camarilla H3/L3 levels.
# Designed for 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Combines breakout in trending regimes with mean reversion in ranging regimes for dual-market effectiveness.

name = "12h_Camarilla_R3S3_ADXRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and ADX calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * camarilla_range
    s3_1d = close_1d - 1.1 * camarilla_range
    h3_1d = close_1d + 1.1 * camarilla_range / 2  # H3 = close + 1.1*(high-low)/2
    l3_1d = close_1d - 1.1 * camarilla_range / 2  # L3 = close - 1.1*(high-low)/2
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 1d ADX for regime filtering
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    
    # Initialize smoothed values
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # First ATR is simple average of first 'atr_period' TR values
    if len(tr) >= atr_period + 1:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        plus_dm_smooth[atr_period] = np.nanmean(plus_dm[1:atr_period+1])
        minus_dm_smooth[atr_period] = np.nanmean(minus_dm[1:atr_period+1])
        
        # Subsequent values using Wilder's smoothing
        for i in range(atr_period + 1, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / atr_period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / atr_period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / atr_period) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    valid_atr = ~np.isnan(atr) & (atr != 0)
    plus_di[valid_atr] = 100 * plus_dm_smooth[valid_atr] / atr[valid_atr]
    minus_di[valid_atr] = 100 * minus_dm_smooth[valid_atr] / atr[valid_atr]
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    valid_di = ~np.isnan(di_sum) & (di_sum != 0)
    dx[valid_di] = 100 * np.abs(plus_di[valid_di] - minus_di[valid_di]) / di_sum[valid_di]
    
    # ADX is EMA of DX
    adx = np.full_like(dx, np.nan)
    adx_period = 14
    if len(dx) >= adx_period + 1:
        # First ADX is simple average of first 'adx_period' DX values
        adx[adx_period] = np.nanmean(dx[1:adx_period+1])
        # Subsequent values using EMA
        for i in range(adx_period + 1, len(dx)):
            if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                adx[i] = (dx[i] - adx[i-1]) * (2.0 / (adx_period + 1)) + adx[i-1]
            else:
                adx[i] = adx[i-1]
    
    # Align ADX to 12h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_value = adx_1d_aligned[i]
        is_trending = adx_value > 25
        
        if position == 0:
            if is_trending:
                # Trending market: breakout strategy
                # Long conditions: price breaks above R3 AND volume spike
                if close[i] > r3_1d_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: price breaks below S3 AND volume spike
                elif close[i] < s3_1d_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging market: mean reversion at H3/L3
                # Long conditions: price crosses above L3 AND volume spike
                if close[i] > l3_1d_aligned[i] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                # Short conditions: price crosses below H3 AND volume spike
                elif close[i] < h3_1d_aligned[i] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long conditions
            if is_trending:
                # In trending market: exit when price re-enters Camarilla range (S3-R3)
                if close[i] >= s3_1d_aligned[i] and close[i] <= r3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging market: exit when price reaches opposite H3 level or midpoint
                if close[i] >= h3_1d_aligned[i] or close[i] <= (h3_1d_aligned[i] + l3_1d_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Exit short conditions
            if is_trending:
                # In trending market: exit when price re-enters Camarilla range (S3-R3)
                if close[i] >= s3_1d_aligned[i] and close[i] <= r3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging market: exit when price reaches opposite L3 level or midpoint
                if close[i] <= l3_1d_aligned[i] or close[i] >= (h3_1d_aligned[i] + l3_1d_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals