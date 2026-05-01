#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation (>2.0x 20-bar volume MA) and 1w ADX(14) trend filter
# Donchian channels provide robust price channels for breakout trading. Volume spike confirms institutional participation.
# 1w ADX > 25 ensures we only trade in strong trending markets (weekly timeframe), reducing false breakouts.
# Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# Works in bull (breakouts with volume) and bear (trend continuation after pullbacks to channel).
# Uses 1d primary timeframe with 1w HTF for trend filter as specified in experiment.

name = "1d_Donchian20_Breakout_VolumeSpike_1wADX25_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 1w ADX(14) calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values using Wilder's smoothing
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
    
    atr_1w = _wilder_smooth(tr, 14)
    plus_di_1w = 100 * _wilder_smooth(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * _wilder_smooth(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = _wilder_smooth(dx_1w, 14)
    
    # Align 1w ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Donchian(20) channels on 1d data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 14*3 for ADX + 20 for Donchian + volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions (using prior bar channels to avoid look-ahead)
        breakout_up = curr_close > donchian_high[i-1]  # Break above upper channel
        breakout_down = curr_close < donchian_low[i-1]  # Break below lower channel
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike[i]
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, volume spike, strong trend
            if breakout_up and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, volume spike, strong trend
            elif breakout_down and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or weak trend
            if curr_close < donchian_low[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or weak trend
            if curr_close > donchian_high[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals