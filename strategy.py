#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + chop regime filter
# Long when price breaks above Donchian(20) high with 1d EMA34 uptrend, volume spike, and chop < 61.8 (trending)
# Short when price breaks below Donchian(20) low with 1d EMA34 downtrend, volume spike, and chop < 61.8
# Designed for 12-37 trades/year on 12h to minimize fee drag while capturing strong trends in both bull and bear markets.
# Uses chop filter to avoid whipsaws in ranging markets and focus on trending conditions.

name = "12h_Donchian20_1dEMA34_Trend_Volume_ChopFilter"
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
    
    # Get 1d data for HTF trend filter and chop calculation - ONCE before loop
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
    
    # Calculate Choppiness Index (CHOP) on 1d data to filter ranging markets
    # CHOP = 100 * log10(sum(ATR(14) over 14 periods) / (log10(highest high - lowest low over 14 periods)))
    tr1 = pd.Series(high_1d).rolling(2).max().values - pd.Series(low_1d).rolling(2).min().values
    tr2 = abs(pd.Series(high_1d).rolling(2).max().values - pd.Series(close_1d).rolling(2).shift(1).values)
    tr3 = abs(pd.Series(low_1d).rolling(2).min().values - pd.Series(close_1d).rolling(2).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = highest_high_14 - lowest_low_14
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned < 61.8  # Trending regime when CHOP < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND 1d uptrend AND volume spike AND trending regime
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i] and
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND 1d downtrend AND volume spike AND trending regime
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i] and
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band OR 1d trend turns down OR chop > 61.8 (ranging)
            if (close[i] < donchian_lower[i] or 
                close[i] < ema_34_aligned[i] or
                not chop_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band OR 1d trend turns up OR chop > 61.8 (ranging)
            if (close[i] > donchian_upper[i] or 
                close[i] > ema_34_aligned[i] or
                not chop_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals