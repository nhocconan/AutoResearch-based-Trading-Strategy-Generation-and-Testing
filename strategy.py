#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA21 trend + volume spike (1.8x) → discrete 0.30 position
# Donchian breakout captures momentum; 1d HMA21 filters trend direction; volume spike confirms institutional interest
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue with trend filter)
# Discrete sizing (0.30) controls drawdown; volume filter reduces overtrading to ~20-40 trades/year on 4h

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # === 1d Indicator: HMA21 ===
    close_1d = df_1d['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = wma(close_1d, half_n)
    wma_full = wma(close_1d, 21)
    wma_diff = 2 * wma_half - wma_full
    # Pad to original length
    wma_diff_padded = np.full_like(close_1d, np.nan)
    wma_diff_padded[half_n-1:len(wma_diff)+half_n-1] = wma_diff
    hma_21_1d = wma(wma_diff_padded, sqrt_n)
    hma_21_1d_padded = np.full_like(close_1d, np.nan)
    hma_21_1d_padded[sqrt_n-1:len(hma_21_1d)+sqrt_n-1] = hma_21_1d
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d_padded)
    
    # === 4h Donchian(20) ===
    def donchian_channels(high, low, n):
        upper = pd.Series(high).rolling(window=n, min_periods=n).max().values
        lower = pd.Series(low).rolling(window=n, min_periods=n).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(21, 20) + 5  # HMA21 + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (close > upper_20)
        # 2. 1d HMA21 uptrend (close > HMA21)
        # 3. Volume confirmation
        if (close[i] > upper_20[i]) and \
           (close[i] > hma_21_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (close < lower_20)
        # 2. 1d HMA21 downtrend (close < HMA21)
        # 3. Volume confirmation
        elif (close[i] < lower_20[i]) and \
             (close[i] < hma_21_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1dHMA21_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0