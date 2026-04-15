#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Uses 1w EMA34 for trend bias and 1d Donchian channels for entry timing.
# Includes volume filter (current volume > 1.3x 20-bar SMA) to avoid low-momentum breakouts.
# Designed for very low trade frequency (10-25/year) to minimize fee drag in choppy markets.
# Works in bull/bear: 1w EMA34 identifies primary trend, Donchian breakouts capture momentum,
# volume confirmation ensures conviction, and trend filter avoids counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channels (20) ===
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # Upper band: 20-period high, lower band: 20-period low
    upper_20 = high_1d.rolling(window=20, min_periods=20).max().values
    lower_20 = low_1d.rolling(window=20, min_periods=20).min().values
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === 1w Indicators: Trend Filter (EMA34) ===
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 1.3x 20-period SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper Donchian channel (20-bar high)
        # 2. 1w price above EMA34 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (close[i] > upper_20_aligned[i] and
            close[i] > ema_34_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower Donchian channel (20-bar low)
        # 2. 1w price below EMA34 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (close[i] < lower_20_aligned[i] and
              close[i] < ema_34_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_EMA34_VolFilter_1w_v1"
timeframe = "1d"
leverage = 1.0