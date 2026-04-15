#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and chop regime filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + chop < 61.8 (trending)
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + chop < 61.8 (trending)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Donchian channels provide objective breakout levels. Volume confirms conviction. Chop filter ensures we trade trends, not ranges.
# Works in bull markets (upward breakouts) and bear markets (downward breakdowns) by requiring trending regime.

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
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Donchian Channels (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels: upper = rolling max(high, 20), lower = rolling min(low, 20)
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Volume SMA (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1d Indicator: Choppiness Index (14-period) ===
    # Chop = 100 * log10(sum(TR,14) / (log10(ATR(14)) * 14))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14)
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of TR over 14 periods
    tr_sum = np.zeros_like(tr)
    for i in range(atr_period-1, len(tr)):
        if i == atr_period-1:
            tr_sum[i] = np.sum(tr[:atr_period])
        else:
            tr_sum[i] = tr_sum[i-1] - tr[i-atr_period] + tr[i]
    
    # Choppiness Index
    chop = np.zeros_like(tr)
    for i in range(atr_period-1, len(tr)):
        if atr[i] > 0 and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (np.log10(atr[i]) * atr_period)) / np.log10(atr_period)
        else:
            chop[i] = 50.0  # neutral value
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_aligned[i] * 1.5)
        
        # Chop filter: trending regime (chop < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Donchian high
        # 2. Volume confirmation
        # 3. Trending regime
        if (close[i] > donchian_high_aligned[i]) and \
           vol_confirm and trending_regime:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Donchian low
        # 2. Volume confirmation
        # 3. Trending regime
        elif (close[i] < donchian_low_aligned[i]) and \
             vol_confirm and trending_regime:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_Volume_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0