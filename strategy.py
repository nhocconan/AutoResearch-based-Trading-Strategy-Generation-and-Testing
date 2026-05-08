#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with weekly trend filter and volume confirmation.
# Uses weekly Donchian channels (20-period) for breakout signals, aligned with weekly trend (price > weekly EMA20).
# Volume confirmation requires current volume > 1.5x 20-period average.
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation.
# Target: 10-25 trades/year to minimize fee drift on daily timeframe.

name = "1d_Donchian_Breakout_1wTrend_Volume"
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
    
    # Weekly trend: EMA20
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly high-low for Donchian calculation (based on previous weekly candle)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian channels (20-period high/low of weekly data)
    # Using rolling window on weekly high/low
    dh_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_1w, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1w, dl_20)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(dh_20_aligned[i]) or 
            np.isnan(dl_20_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian high + uptrend (price > weekly EMA20) + volume spike
            long_cond = (close[i] > dh_20_aligned[i]) and \
                        (close[i] > ema_20_1w_aligned[i]) and \
                        volume_spike[i]
            # Short: break below weekly Donchian low + downtrend (price < weekly EMA20) + volume spike
            short_cond = (close[i] < dl_20_aligned[i]) and \
                         (close[i] < ema_20_1w_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below weekly Donchian low (mean reversion)
            if close[i] < dl_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above weekly Donchian high (mean reversion)
            if close[i] > dh_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals