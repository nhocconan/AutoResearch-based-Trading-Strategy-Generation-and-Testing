#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme with 1d ADX Regime Filter and Volume Confirmation
- Williams %R(14) identifies overbought/oversold extremes: < -80 = oversold, > -20 = overbought
- 1d ADX(14) defines trend regime: ADX > 25 = trending (fade extremes), ADX < 20 = ranging (mean revert)
- In trending markets (ADX>25): short when %R > -20 and price < 1d EMA50, long when %R < -80 and price > 1d EMA50
- In ranging markets (ADX<20): long when %R < -80, short when %R > -20 (pure mean reversion)
- Volume confirmation (> 1.5x 24-period MA) reduces false signals
- Designed for 12h timeframe to capture medium-term reversals with controlled frequency (target: 12-37 trades/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d ADX(14) for regime filter
    # ADX calculation: +DM, -DM, TR, then DX, then smoothed ADX
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - high_1d[:-1]), 
                               np.abs(low_1d[1:] - low_1d[:-1])))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Avoid division by zero
    divisor = tr_smooth.copy()
    divisor[divisor == 0] = 1e-10
    
    plus_di = 100 * plus_dm_smooth / divisor
    minus_di = 100 * minus_dm_smooth / divisor
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = wilder_smooth(dx, period)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 24, 28)  # Williams %R, EMA50, Vol MA, ADX (with smoothing)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine market regime based on ADX
            is_trending = adx_aligned[i] > 25
            is_ranging = adx_aligned[i] < 20
            
            # Volume confirmation
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            if vol_confirm:
                if is_trending:
                    # In trending market: fade extremes only if aligned with trend
                    if williams_r_aligned[i] < -80 and close[i] > ema_50_1d_aligned[i]:
                        # Oversold and above EMA50 = long in uptrend
                        signals[i] = 0.25
                        position = 1
                    elif williams_r_aligned[i] > -20 and close[i] < ema_50_1d_aligned[i]:
                        # Overbought and below EMA50 = short in downtrend
                        signals[i] = -0.25
                        position = -1
                elif is_ranging:
                    # In ranging market: pure mean reversion at extremes
                    if williams_r_aligned[i] < -80:
                        # Oversold = long
                        signals[i] = 0.25
                        position = 1
                    elif williams_r_aligned[i] > -20:
                        # Overbought = short
                        signals[i] = -0.25
                        position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to overbought OR crosses below EMA50 in trending market
                if williams_r_aligned[i] > -20:
                    exit_signal = True
                elif adx_aligned[i] > 25 and close[i] < ema_50_1d_aligned[i]:
                    # In trending market, exit if price breaks below EMA50
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R returns to oversold OR crosses above EMA50 in trending market
                if williams_r_aligned[i] < -80:
                    exit_signal = True
                elif adx_aligned[i] > 25 and close[i] > ema_50_1d_aligned[i]:
                    # In trending market, exit if price breaks above EMA50
                    exit_signal = True
            
            # Also exit if volatility spikes (potential false signal)
            if volume[i] > 3.0 * vol_ma[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dADX_Regime_VolumeConfirm"
timeframe = "12h"
leverage = 1.0