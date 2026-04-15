#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Uses 4h Donchian channels for breakout entries, filtered by 12h EMA trend and volume spike.
# Choppiness index regime filter avoids trades in sideways markets, reducing whipsaw.
# Designed for low trade frequency (20-40/year) to minimize fee drag. Works in bull/bear:
# - Bull: long Donchian upper breakouts with volume
# - Bear: short Donchian lower breakouts with volume
# - Range: choppiness filter prevents false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h and 12h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    if len(df_4h) < 50 or len(df_12h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channels (20) ===
    # Donchian Upper = highest high over 20 periods
    # Donchian Lower = lowest low over 20 periods
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels on 4h data
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # === 12h Indicators: EMA(34) Trend Filter ===
    # 12h EMA(34) for trend bias
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h Indicators: Choppiness Index (14) ===
    # Choppiness Index = 100 * log10(sum(ATR) / (log10(n) * (highest high - lowest low)))
    # Simplified: CHOP = 100 * log10( sum(TR) / (log10(n) * (HHV - LLV)) )
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop_12h = 100 * np.log10(tr_sum / (np.log10(14) * (hh_14 - ll_14)))
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only (reduces noise trades)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian Upper (20-period high)
        # 2. 12h price above EMA34 (bullish trend bias)
        # 3. Volume confirmation
        # 4. Chop < 61.8 (trending market, not choppy)
        if (close[i] > donchian_upper_aligned[i] and
            close[i] > ema_34_12h_aligned[i] and
            vol_confirm and
            chop_12h_aligned[i] < 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian Lower (20-period low)
        # 2. 12h price below EMA34 (bearish trend bias)
        # 3. Volume confirmation
        # 4. Chop < 61.8 (trending market, not choppy)
        elif (close[i] < donchian_lower_aligned[i] and
              close[i] < ema_34_12h_aligned[i] and
              vol_confirm and
              chop_12h_aligned[i] < 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_EMA34_Vol_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0