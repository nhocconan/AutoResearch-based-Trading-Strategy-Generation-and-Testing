#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h KAMA trend + volume confirmation.
# Goes long when price breaks above 4h Donchian upper band with volume > 2x average and 12h KAMA rising.
# Goes short when price breaks below 4h Donchian lower band with volume > 2x average and 12h KAMA falling.
# Uses ATR(14) for stoploss (2x ATR). Target: 75-200 total trades over 4 years (19-50/year).
# Designed to work in both bull and bear markets by requiring volume confirmation and trend alignment.

name = "4h_donchian20_kama12_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 12h KAMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate ER (Efficiency Ratio) for KAMA
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0)
    # Correct calculation: ER = |change| / sum(|volatility|) over period
    change_pd = pd.Series(change)
    volatility_pd = pd.Series(np.abs(np.diff(close_12h, prepend=close_12h[0])))
    er = change_pd.rolling(window=10, min_periods=1).sum() / volatility_pd.rolling(window=10, min_periods=1).sum()
    er = er.fillna(0).values
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_slope = kama - np.roll(kama, 1)
    kama_slope[0] = 0
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    kama_slope_aligned = align_htf_to_ltf(prices, df_12h, kama_slope)
    
    # Volume filter (2x average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(kama_aligned[i]) or np.isnan(kama_slope_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower or KAMA turns down
            elif close[i] < donchian_low[i] or kama_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper or KAMA turns up
            elif close[i] > donchian_high[i] or kama_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and KAMA trend alignment
            if vol_filter[i]:
                # Long breakout: price breaks above Donchian upper with volume and rising KAMA
                if close[i] > donchian_high[i] and kama_slope_aligned[i] > 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian lower with volume and falling KAMA
                elif close[i] < donchian_low[i] and kama_slope_aligned[i] < 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals