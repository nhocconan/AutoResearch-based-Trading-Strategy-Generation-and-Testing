#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1d price channel breakouts for entry signals, weekly EMA50 for trend direction,
# and volume spikes (2x 20-period average) to confirm breakouts. Designed to work in both
# bull and bear markets by following the 1w trend while entering on 1d breakouts.
# Target: 10-20 trades/year to minimize fee decay while capturing trend continuation moves.
# Focus on BTC/ETH as primary assets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_channel = np.full(len(high_1d), np.nan)
    lower_channel = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        upper_channel[i] = np.max(high_1d[i-20:i])
        lower_channel[i] = np.min(low_1d[i-20:i])
    
    # Calculate 50-period EMA on 1w for trend
    close_1w = df_1w['close'].values
    ema_len = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1w[ema_len-1] = np.mean(close_1w[:ema_len])
        for i in range(ema_len, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 1d timeframe (then to lower timeframe via index alignment)
    upper_aligned_1d = align_htf_to_ltf(df_1d['close'], df_1d, upper_channel)
    lower_aligned_1d = align_htf_to_ltf(df_1d['close'], df_1d, lower_channel)
    ema_1w_aligned_1d = align_htf_to_ltf(df_1d['close'], df_1d, ema_1w)
    
    # Further align to the actual trading timeframe (1d data aligned to lower TF)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_aligned_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_aligned_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_1w_aligned_1d)
    
    # Calculate 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(30, 50) + 20  # Donchian needs 20, EMA50 needs 50, volume needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: Price breaks above upper Donchian with uptrend and volume
            if price > upper_aligned[i] and price > ema_1w_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Donchian with downtrend and volume
            elif price < lower_aligned[i] and price < ema_1w_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below lower Donchian or trend reversal
            if price < lower_aligned[i] or price < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above upper Donchian or trend reversal
            if price > upper_aligned[i] or price > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_DonchianBreakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0