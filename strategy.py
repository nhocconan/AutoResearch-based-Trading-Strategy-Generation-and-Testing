#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 indicates strong trend (use trend-following logic), ADX < 20 indicates range (use mean reversion)
# Volume confirmation filters low-momentum breakouts/breakdowns
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets via bull power expansion in uptrends, bear markets via bear power expansion in downtrends

name = "6h_ElderRay_1dADXRegime_Volume"
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime detection
    # ADX calculation requires +DI, -DI, and true range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # +DI and -DI
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilder_smooth(dx, period)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA and EMA13
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])  # Moderate volume spike
        
        # Regime detection
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        if position == 0:
            # Long conditions
            if is_trending:
                # In trend: follow bull power expansion
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_spike:
                    signals[i] = 0.25
                    position = 1
            elif is_ranging:
                # In range: mean reversion from bear power
                if bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and volume_spike:
                    signals[i] = 0.25
                    position = 1
            
            # Short conditions
            if is_trending:
                # In trend: follow bear power expansion
                if bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and volume_spike:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # In range: mean reversion from bull power
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: power deteriorates or regime changes against position
            if (is_trending and bull_power[i] < 0) or (is_ranging and bull_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: power deteriorates or regime changes against position
            if (is_trending and bear_power[i] > 0) or (is_ranging and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals