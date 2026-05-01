#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d for signal direction and 1h for entry timing.
# Direction: 4h Donchian(20) breakout + volume confirmation + 1d ADX(25) trend filter.
# Entry timing: 1h pullback to 20 EMA in direction of 4h breakout.
# Uses discrete sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull (breakouts with volume) and bear (trend continuation after pullbacks to EMA).
# Session filter: 08-20 UTC to avoid low-liquidity hours.

name = "1h_Donchian20_Breakout_VolumeSpike_1dADX25_Trend_EMA20_Pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: current volume > 2.0 * 20-period average volume
    volume_4h = df_4h['volume'].values
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (volume_ma_20_4h * 2.0)
    
    # Align 4h indicators to 1h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h.astype(float))
    
    # 1d HTF data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX(14) calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
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
    
    # Wilder smoothing
    def _wilder_smooth(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(x[1:period])
        # Subsequent values are Wilder smoothing
        for i in range(period, len(x)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr_1d = _wilder_smooth(tr, 14)
    plus_di_1d = 100 * _wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * _wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = _wilder_smooth(dx_1d, 14)
    
    # Align 1d ADX to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1h EMA(20) for pullback entries
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 20 for Donchian + 20 for EMA + buffer
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(volume_spike_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # 4h Donchian breakout conditions (using prior bar channels to avoid look-ahead)
        breakout_up = curr_close > donchian_high_4h_aligned[i-1]  # Break above upper channel
        breakout_down = curr_close < donchian_low_4h_aligned[i-1]  # Break below lower channel
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike_4h_aligned[i] > 0.5  # Convert back to boolean
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: 4h breakout up, volume spike, strong trend, and 1h pullback to EMA20
            if breakout_up and vol_spike and strong_trend and curr_close <= ema_20[i] * 1.005:
                signals[i] = 0.20
                position = 1
            # Short: 4h breakout down, volume spike, strong trend, and 1h pullback to EMA20
            elif breakout_down and vol_spike and strong_trend and curr_close >= ema_20[i] * 0.995:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 4h Donchian breakdown or weak trend
            if curr_close < donchian_low_4h_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on 4h Donchian breakout or weak trend
            if curr_close > donchian_high_4h_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals