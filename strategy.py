#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout (20-period) with 1w EMA(34) trend filter and volume confirmation.
# Uses 1w EMA(34) for long-term trend bias and Donchian breakout for entry timing.
# Includes volume filter (current volume > 1.5x 20-bar SMA) to avoid low-momentum breakouts.
# Designed for very low trade frequency (7-25/year) to minimize fee drag in choppy markets.
# Works in bull/bear: 1w EMA avoids counter-trend trades, Donchian captures breakouts with momentum.

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
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(34) for long-term trend bias
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (20-period breakout)
        # 2. 1w price above EMA34 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i] and
            close[i] > ema_34_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (20-period breakdown)
        # 2. 1w price below EMA34 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i] and
              close[i] < ema_34_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_EMA34_VolFilter_v1"
timeframe = "1d"
leverage = 1.0