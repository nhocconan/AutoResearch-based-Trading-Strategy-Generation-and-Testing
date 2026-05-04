#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + chop regime filter
# Long when price breaks above Donchian(20) high with 1d uptrend, volume spike, and choppy regime (CHOP > 61.8)
# Short when price breaks below Donchian(20) low with 1d downtrend, volume spike, and choppy regime (CHOP > 61.8)
# Designed for 12-37 trades/year on 12h to minimize fee drag while capturing mean reversion in ranging markets.
# Uses chop regime to avoid false breakouts in strong trends, improving performance in bear markets like 2025.

name = "12h_Donchian20_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and chop regime - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    # Calculate Choppiness Index (CHOP) on 1d data
    # CHOP = 100 * log10(sum(ATR(1) over 14 periods) / (log10(highest high - lowest low over 14 periods)))
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 / (highest_high - lowest_low)) / np.log10(14)
    chop[np.isnan(chop) | (highest_high - lowest_low) == 0] = 50  # default to neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Chop regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion/breakout fade)
    chop_range = chop_aligned > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND 1d uptrend AND volume spike AND choppy regime
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i] and
                chop_range[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND 1d downtrend AND volume spike AND choppy regime
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i] and
                  chop_range[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band OR 1d trend turns down OR chop regime ends
            if (close[i] < donchian_lower[i] or 
                close[i] < ema_34_aligned[i] or
                not chop_range[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band OR 1d trend turns up OR chop regime ends
            if (close[i] > donchian_upper[i] or 
                close[i] > ema_34_aligned[i] or
                not chop_range[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals