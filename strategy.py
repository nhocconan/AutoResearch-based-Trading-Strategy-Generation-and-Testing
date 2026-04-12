#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R extreme reversal with 1w EMA200 trend filter and volume confirmation
    # Uses 1w for dominant trend direction (avoid counter-trend trades), 12h for entry timing
    # Williams %R < -80 = oversold (long), > -20 = overbought (short) on 12h
    # Volume spike (>1.5x 24-period average) confirms institutional participation
    # Target: 12-30 trades/year (48-120 total over 4 years) to minimize fee drag
    # Only trades with the dominant 1w trend to avoid whipsaws in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 12h data for Williams %R and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 1w EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Williams %R on 12h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    lookback = 14
    highest_high = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero (when HH == LL)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume spike on 12h (>1.5x 24-period average)
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(24, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-24:i])
    volume_spike_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Align all indicators to LTF (prices timeframe)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike_12h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # 1w trend filter
        bullish_trend = close[i] > ema200_1w_aligned[i]
        bearish_trend = close[i] < ema200_1w_aligned[i]
        
        # Entry logic: Extreme Williams %R + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike_12h_aligned[i]
        short_entry = overbought and bearish_trend and volume_spike_12h_aligned[i]
        
        # Exit logic: Williams %R returns to neutral zone (-50) or trend reversal
        long_exit = williams_r_aligned[i] > -50 or not bullish_trend
        short_exit = williams_r_aligned[i] < -50 or not bearish_trend
        
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

name = "12h_1w_williamsr_extreme_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0