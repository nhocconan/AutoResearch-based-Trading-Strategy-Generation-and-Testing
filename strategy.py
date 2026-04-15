#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA34 trend filter with daily Donchian(20) breakout and volume confirmation.
# Uses weekly EMA34 for trend bias (avoids counter-trend trades in bear markets like 2022 and 2025+).
# Daily Donchian(20) breakout captures institutional level reactions with volume confirmation.
# Designed for very low trade frequency (7-25/year) to minimize fee drag. Works in bull/bear:
# - Weekly EMA34 ensures we only trade with the dominant trend
# - Donchian breakouts with volume filter reduce false signals
# - Discrete position sizing (0.25) controls drawdown during crashes

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA(34) for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Indicators: Donchian(20) channels ===
    # Using rolling window with min_periods
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_sma_20 * 1.5)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian(20) upper channel
        # 2. Weekly price above EMA34 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > high_max_20[i] and
            close[i] > ema_34_1w_aligned[i] and
            vol_confirm[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian(20) lower channel
        # 2. Weekly price below EMA34 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < low_min_20[i] and
              close[i] < ema_34_1w_aligned[i] and
              vol_confirm[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_EMA34_1w_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0