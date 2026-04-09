#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 1d strategy using Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trending). Enters long on upper band breakout in trending/regime, short on lower band breakout. Uses weekly EMA(20) as HTF trend filter: only long when price > weekly EMA20, short when price < weekly EMA20. Discrete position sizing (0.25) to limit fee drag. Target: 20-80 trades over 4 years (5-20/year) to avoid fee drag and work in both bull and bear markets by following breakouts aligned with weekly trend and volatility regime.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    tr1 = np.subtract(high, low)
    tr2 = np.subtract(np.abs(high), np.roll(close, 1))
    tr3 = np.subtract(np.abs(low), np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean()
    return atr.values

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    atr = calculate_atr(high, low, close, 1)
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

name = "1d_donchian_breakout_volume_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    chop = calculate_choppiness(high, low, close, 14)
    
    # Weekly HTF trend filter: 20-period EMA on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(chop[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and weekly trend alignment
            if volume_confirmed:
                # Weekly bullish trend: price above weekly EMA20
                weekly_bullish = close[i] > ema_20_1w_aligned[i]
                # Weekly bearish trend: price below weekly EMA20
                weekly_bearish = close[i] < ema_20_1w_aligned[i]
                
                # Regime filter: only trade in trending markets (CHOP < 61.8)
                trending_regime = chop[i] < 61.8
                
                # Long: price breaks above Donchian upper band with volume, weekly trend, and trending regime
                if close[i] > highest_high[i] and weekly_bullish and trending_regime:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band with volume, weekly trend, and trending regime
                elif close[i] < lowest_low[i] and weekly_bearish and trending_regime:
                    position = -1
                    signals[i] = -0.25
    
    return signals