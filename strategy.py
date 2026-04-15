#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA200 trend filter + Donchian(20) breakout + volume confirmation.
# Uses weekly EMA200 for trend bias (avoid counter-trend trades), daily Donchian breakout for entry,
# and volume spike for confirmation. Designed for low trade frequency (10-25/year) to minimize fee drag.
# Works in bull/bear: weekly EMA200 avoids counter-trend trades, Donchian breakouts capture
# sustained momentum with volume confirmation. Position size 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA200 for trend filter ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    # Donchian high: max(high, lookback=20)
    # Donchian low: min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period volume SMA
    vol_series = pd.Series(volume)
    vol_sma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_sma_20 * 2.0)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200  # for EMA200
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian(20) high
        # 2. 1w price above EMA200 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i] and
            close[i] > ema_200_1w_aligned[i] and
            vol_confirm[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian(20) low
        # 2. 1w price below EMA200 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i] and
              close[i] < ema_200_1w_aligned[i] and
              vol_confirm[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_EMA200_1w_VolFilter_v1"
timeframe = "1d"
leverage = 1.0