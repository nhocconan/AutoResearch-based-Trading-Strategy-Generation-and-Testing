#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation
# Uses 1d primary timeframe to target 30-100 trades over 4 years (7-25/year).
# Donchian channel from 20-period provides clear breakout levels. 
# 1w HMA(21) filters for higher timeframe trend alignment.
# Volume spike (2.0x 20-period average) confirms breakout validity.
# Discrete sizing 0.25 minimizes fee churn. Works in bull via breakout longs,
# in bear via breakout shorts with trend filter.

name = "1d_Donchian20_Breakout_1wHMA21_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w HMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    wma_half = np.array([wma(close_1w[i:i+half_len], half_len) if i+half_len <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    wma_full = np.array([wma(close_1w[i:i+21], 21) if i+21 <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    hma_1w_raw = 2 * wma_half - wma_full
    hma_1w = np.array([wma(hma_1w_raw[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(hma_1w_raw) else np.nan 
                       for i in range(len(hma_1w_raw))])
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 21)  # warmup for Donchian and HMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high_roll = high_roll[i]
        curr_low_roll = low_roll[i]
        curr_hma = hma_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above upper Donchian with 1w HMA uptrend
                if curr_close > curr_high_roll and curr_close > curr_hma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lower Donchian with 1w HMA downtrend
                elif curr_close < curr_low_roll and curr_close < curr_hma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below lower Donchian (mean reversion)
            if curr_close < curr_low_roll:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above upper Donchian (mean reversion)
            if curr_close > curr_high_roll:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals