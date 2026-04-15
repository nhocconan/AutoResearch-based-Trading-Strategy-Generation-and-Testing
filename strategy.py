#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Uses 1d Donchian upper/lower bands for breakout signals, filtered by 1w EMA200 for long-term trend bias.
# Volume confirmation ensures breakouts are supported by participation. Designed for low trade frequency
# (12-37 trades/year) to minimize fee drag. Works in bull/bear: 1w EMA200 avoids counter-trend trades,
# Donchian breakouts capture sustained momentum after consolidation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian(20) Channels ===
    # Donchian upper/lower = rolling max(high)/min(low) over 20 periods
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(200) for long-term trend bias
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only (reduces noise, maintains liquidity)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 50-period volume SMA
        vol_sma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
        vol_confirm = volume[i] > (vol_sma_50[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Donchian(20) upper band
        # 2. 1w price above EMA200 (bullish long-term trend)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i] and
            close[i] > ema_200_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Donchian(20) lower band
        # 2. 1w price below EMA200 (bearish long-term trend)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i] and
              close[i] < ema_200_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_EMA200_1w_VolFilter_v1"
timeframe = "12h"
leverage = 1.0