#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 12h/1d regime filter
    # Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13
    # Regime: 12h ADX > 25 = trending (trade Elder Ray signals), ADX < 20 = range (fade extremes)
    # Volume confirmation: current 6h volume > 1.5x 20-period average
    # Target: 12-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 (Elder Ray base)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Get 12h data for regime filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14)
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
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value: simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: Wilder smoothing
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]):
                        result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr = wilder_smooth(tr, period)
        smoothed_plus_dm = wilder_smooth(plus_dm, period)
        smoothed_minus_dm = wilder_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        # Wilder smoothing for ADX
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 12h ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Elder Ray signals: Bull Power > 0 = buying pressure, Bear Power < 0 = selling pressure
        # In trending regime: follow Elder Ray (bull power > 0 = long, bear power < 0 = short)
        # In ranging regime: fade extremes (bull power < 0 = long, bear power > 0 = short)
        if trending and vol_ok:
            long_signal = bull_power_aligned[i] > 0
            short_signal = bear_power_aligned[i] < 0
        elif ranging and vol_ok:
            long_signal = bear_power_aligned[i] > 0  # fade selling pressure
            short_signal = bull_power_aligned[i] > 0  # fade buying pressure
        else:
            long_signal = False
            short_signal = False
        
        # Exit conditions: opposing Elder Ray signal or regime change
        long_exit = bear_power_aligned[i] < 0  # selling pressure appears
        short_exit = bull_power_aligned[i] > 0   # buying pressure appears
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0