#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# ADX > 25 indicates trending market (use Elder Ray signals), ADX < 20 indicates ranging (fade extremes)
# Volume confirmation filters low-conviction moves. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via bull power breakouts and bear markets via bear power breakdowns with ADX filter.

name = "6h_ElderRay_ADX_Regime_Volume"
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
    
    # Get 1d data for ADX regime filter and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime filter (using Welles Wilder's method)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value: simple average
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            return result
        
        tr_smooth = WilderSmoothing(tr, period)
        plus_dm_smooth = WilderSmoothing(plus_dm, period)
        minus_dm_smooth = WilderSmoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = WilderSmoothing(dx, period)
        return adx
    
    # Calculate 1d ADX
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Align 1d EMA13 for Elder Ray calculation
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components on 6h data
    bull_power = high - ema_13_1d_aligned  # High - EMA13
    bear_power = low - ema_13_1d_aligned   # Low - EMA13
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start from 13 to have valid EMA13 values
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        adx_val = adx_1d_aligned[i]
        
        if position == 0:
            # Regime-based entries
            if adx_val > 25:  # Trending market - follow Elder Ray
                # Long: bull power positive and increasing with volume spike
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: bear power negative and decreasing with volume spike
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and volume_spike:
                    signals[i] = -0.25
                    position = -1
            elif adx_val < 20:  # Ranging market - fade extremes
                # Long: bear power extremely negative (oversold) with volume spike
                if bear_power[i] < -np.std(bear_power[max(0, i-50):i]) * 2.0 and volume_spike:
                    signals[i] = 0.20
                    position = 1
                # Short: bull power extremely positive (overbought) with volume spike
                elif bull_power[i] > np.std(bull_power[max(0, i-50):i]) * 2.0 and volume_spike:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: bear power turns positive or loses momentum
            if bear_power[i] > 0 or (adx_val > 25 and bull_power[i] < bull_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power turns negative or loses momentum
            if bull_power[i] < 0 or (adx_val > 25 and bear_power[i] > bear_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals