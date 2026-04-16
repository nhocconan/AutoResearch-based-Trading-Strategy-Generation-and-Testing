#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d ADX trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold bounce) + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Short when Williams %R(14) crosses below -20 (overbought rejection) + 1d ADX > 25 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Williams %R captures mean reversion in extended moves, ADX filters for trending environments to avoid chop,
# Volume confirmation ensures participation. Designed for 12-25 trades/year on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX(14) ===
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
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if period < len(data):
            result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr_1d = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = WilderSmoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 12h Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100, -50)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 14, 20) + 5  # ADX + Williams %R + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Williams %R cross detection (using previous bar)
        prev_williams_r = williams_r[i-1]
        curr_williams_r = williams_r[i]
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (oversold bounce)
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        if (prev_williams_r <= -80 and curr_williams_r > -80) and \
           (adx_1d_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (overbought rejection)
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        elif (prev_williams_r >= -20 and curr_williams_r < -20) and \
             (adx_1d_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR14_1dADX_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0