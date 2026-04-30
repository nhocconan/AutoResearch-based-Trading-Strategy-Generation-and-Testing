#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume spike
# Uses 4h primary timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Donchian breakouts capture momentum moves. 12h HMA(21) filters for trend direction
# to avoid counter-trend trades. Volume spike (2.0x 20-period average) confirms validity.
# Discrete sizing 0.25 minimizes fee churn. Works in bull via breakout longs with uptrend,
# in bear via breakout shorts with downtrend. Includes ATR-based stoploss.

name = "4h_Donchian20_Breakout_12hHMA21_VolumeSpike_v2"
timeframe = "4h"
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
    
    # Calculate 12h HMA(21) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 21)  # warmup for Donchian and HMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_hma = hma_12h_aligned[i]
        curr_highest = highest_20[i]
        curr_lowest = lowest_20[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above Donchian upper with uptrend
                if curr_close > curr_highest and curr_close > curr_hma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below Donchian lower with downtrend
                elif curr_close < curr_lowest and curr_close < curr_hma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2*ATR below entry
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian lower
            elif curr_close < curr_lowest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2*ATR above entry
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian upper
            elif curr_close > curr_highest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan, dtype=np.float64)
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    wma_half = wma(values, half_period)
    wma_full = wma(values, period)
    
    if len(wma_half) == 0 or len(wma_full) == 0:
        return np.full_like(values, np.nan, dtype=np.float64)
    
    # Align arrays (WMA produces shorter arrays)
    diff = 2 * wma_half[-len(wma_full):] - wma_full
    hma = wma(diff, sqrt_period)
    
    # Create full-length array with NaN padding
    hma_full = np.full_like(values, np.nan, dtype=np.float64)
    start_idx = len(values) - len(hma)
    hma_full[start_idx:] = hma
    return hma_full