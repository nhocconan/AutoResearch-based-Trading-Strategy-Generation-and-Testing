#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    close_1w = df_1w['close'].values
    
    # Calculate daily ATR (14-period)
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr_1d[i]
    
    # Calculate 50-day ATR for volatility regime
    atr_50d = np.zeros_like(atr_1d)
    for i in range(len(atr_1d)):
        if i < 50:
            atr_50d[i] = np.mean(atr_1d[:i+1]) if i > 0 else atr_1d[i]
        else:
            atr_50d[i] = 0.98 * atr_50d[i-1] + 0.02 * atr_1d[i]
    
    # Volatility regime: ATR(14) > 1.5 * ATR(50) indicates volatility expansion
    vol_regime = atr_1d > (1.5 * atr_50d)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average (20-period)
    vol_avg_20d = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if i < 20:
            vol_avg_20d[i] = np.mean(volume_1d[:i+1]) if i > 0 else volume_1d[i]
        else:
            vol_avg_20d[i] = 0.95 * vol_avg_20d[i-1] + 0.05 * volume_1d[i]
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_spike = volume_1d > (2.0 * vol_avg_20d)
    
    # Align indicators to 12h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(vol_regime_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion regime
        high_vol = vol_regime_aligned[i]
        
        # Weekly trend filter
        above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        high_volume = vol_spike_aligned[i]
        
        # Entry conditions
        long_entry = high_vol and above_weekly_ema and high_volume
        short_entry = high_vol and below_weekly_ema and high_volume
        
        # Exit conditions: volatility contraction or volume drop
        exit_long = position == 1 and (not high_vol or not high_volume)
        exit_short = position == -1 and (not high_vol or not high_volume)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_vol_regime_trend_volume"
timeframe = "12h"
leverage = 1.0