#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1w trend filter using 1w EMA89 for primary trend direction,
# 4h Donchian20 breakout for entry momentum, volume confirmation (2.0x), and 1d ADX filter (ADX>20).
# Enters only during 08-20 UTC session to avoid low-liquidity periods.
# Targets 15-35 trades/year (60-140 total) with strict multi-condition entry.
# Uses 1w trend to avoid counter-trend trades in strong trends, ADX to avoid ranging markets.
name = "4h_1w_EMA89_Donchian20_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA89 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Get 1d data for ADX (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    atr_1d = np.zeros(len(high_1d))
    plus_di_1d = np.zeros(len(high_1d))
    minus_di_1d = np.zeros(len(high_1d))
    dx_1d = np.zeros(len(high_1d))
    
    # Initial values
    atr_1d[13] = np.mean(tr[1:14])
    plus_dm_14 = np.sum(plus_dm[1:14])
    minus_dm_14 = np.sum(minus_dm[1:14])
    
    for i in range(14, len(high_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
        plus_di_1d[i] = 100 * ((plus_di_1d[i-1] * 13 + plus_dm[i]) / 14) / atr_1d[i]
        minus_di_1d[i] = 100 * ((minus_di_1d[i-1] * 13 + minus_dm[i]) / 14) / atr_1d[i]
        dx_1d[i] = (abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i])) * 100
    
    # Calculate ADX
    adx_1d = np.zeros(len(high_1d))
    adx_1d[27] = np.mean(dx_1d[14:28])
    for i in range(28, len(high_1d)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for Donchian20 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian channels: 20-period high/low
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 150  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_89_1w_aligned[i]) or np.isnan(high_20_4h_aligned[i]) or 
            np.isnan(low_20_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(adx_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when ADX > 20 (trending market)
        adx_filter = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long: price above 1w EMA89 AND breaks 4h Donchian high with volume and ADX
            if (close[i] > ema_89_1w_aligned[i] and 
                close[i] > high_20_4h_aligned[i] and 
                volume_filter[i] and 
                adx_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA89 AND breaks 4h Donchian low with volume and ADX
            elif (close[i] < ema_89_1w_aligned[i] and 
                  close[i] < low_20_4h_aligned[i] and 
                  volume_filter[i] and 
                  adx_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1w EMA89 or 4h Donchian low
            if close[i] < ema_89_1w_aligned[i] or close[i] < low_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1w EMA89 or 4h Donchian high
            if close[i] > ema_89_1w_aligned[i] or close[i] > high_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals