#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + ADX regime filter
    # Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
    # ADX > 25 indicates strong trend, ADX < 20 indicates ranging/weak trend
    # In strong trend (ADX>25): take Elder Ray signals (bull power >0 for long, bear power >0 for short)
    # In weak trend (ADX<20): fade extreme Elder Ray readings (mean reversion)
    # Volume confirmation: require volume > 1.5 * 20-period average to avoid low-vol breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-37 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # High - EMA13
    bear_power_1d = ema13_1d - low_1d   # EMA13 - Low
    
    # Calculate 1d ADX (Average Directional Index)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr_period = 14
    tr_smoothed = wilders_smoothing(tr, atr_period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, atr_period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, atr_period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    # Align all 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime detection
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        # Elder Ray signals
        bull_signal = bull_power_1d_aligned[i] > 0  # Bullish momentum
        bear_signal = bear_power_1d_aligned[i] > 0  # Bearish momentum
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if strong_trend:
            # In strong trend: follow Elder Ray signals
            long_entry = bull_signal and volume_spike[i]
            short_entry = bear_signal and volume_spike[i]
        elif weak_trend:
            # In weak trend: fade extreme readings (mean reversion)
            # Long when bull power is extremely negative (oversold)
            # Short when bear power is extremely negative (overbought)
            long_entry = (bull_power_1d_aligned[i] < -0.5 * np.std(bull_power_1d_aligned[max(0,i-50):i+1])) and volume_spike[i]
            short_entry = (bear_power_1d_aligned[i] < -0.5 * np.std(bear_power_1d_aligned[max(0,i-50):i+1])) and volume_spike[i]
        
        # Exit logic: opposite signal or regime change to opposite extreme
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            long_exit = (not bull_signal and bear_signal) or (adx_aligned[i] > 30 and bear_power_1d_aligned[i] > bull_power_1d_aligned[i])
        elif position == -1:  # Short position
            short_exit = (not bear_signal and bull_signal) or (adx_aligned[i] > 30 and bull_power_1d_aligned[i] > bear_power_1d_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "6h_1d_elder_ray_adx_regime_v2"
timeframe = "6h"
leverage = 1.0