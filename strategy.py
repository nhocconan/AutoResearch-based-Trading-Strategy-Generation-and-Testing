#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly ADX trend filter and volume confirmation
# Uses weekly ADX(14) > 25 to identify strong trending markets (works in bull/bear via direction)
# Donchian(20) breakouts capture momentum in trending regimes
# Volume spike confirms breakout authenticity and reduces false signals
# Designed for low frequency (50-150 trades over 4 years) to minimize fee drag
# Weekly timeframe provides major trend filter, 6h for execution timing

name = "6h_Donchian20_1wADX25_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on weekly data
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
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1w = wilders_smoothing(tr, period)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, period) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, period) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, period)
    
    # Align ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Donchian(20) on 6h data
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(donchian_window, 20, 14*2)  # Need Donchian20, volume MA20, and ADX warmed up
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions using Donchian channels
        breakout_up = close[i] > donchian_high[i-1]  # Break above upper channel
        breakout_down = close[i] < donchian_low[i-1]  # Break below lower channel
        
        # Trend filter: weekly ADX > 25 indicates strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: upward breakout, volume spike, strong trend
            if breakout_up and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout, volume spike, strong trend
            elif breakout_down and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on trend weakness or price re-enters Donchian channel (below midpoint)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if not strong_trend or close[i] < donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on trend weakness or price re-enters Donchian channel (above midpoint)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if not strong_trend or close[i] > donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals