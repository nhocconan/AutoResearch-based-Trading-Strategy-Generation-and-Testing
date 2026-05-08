#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Uses weekly ADX(14) > 25 to confirm trending market, then enters on daily Donchian breakout
# Volume spike (2x 20-day average) confirms breakout strength
# Designed for low trade frequency: target 30-100 total trades over 4 years
# Works in both bull and bear markets by following weekly trend direction

name = "1d_Donchian_1wTrend_Volume"
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
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.maximum(high_1w[1:], low_1w[:-1]) - np.minimum(low_1w[1:], high_1w[:-1])
    tr1 = np.concatenate([[np.nan], tr1])
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothing
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1w = adx  # weekly ADX values
    
    # Trend direction: weekly price above/below 50-period EMA
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    
    # Align weekly indicators to daily
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1w_aligned[i]
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: strong uptrend (ADX>25 + weekly uptrend) + price breaks above Donchian high + volume spike
            if (adx_val > 25 and weekly_up and 
                close[i] > donch_high and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: strong downtrend (ADX>25 + weekly downtrend) + price breaks below Donchian low + volume spike
            elif (adx_val > 25 and not weekly_up and 
                  close[i] < donch_low and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakens (ADX<20) OR price breaks below Donchian low
            if (adx_val < 20 or close[i] < donch_low):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakens (ADX<20) OR price breaks above Donchian high
            if (adx_val < 20 or close[i] > donch_high):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals