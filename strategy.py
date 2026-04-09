#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v4
# Hypothesis: 4h strategy using Donchian channel breakouts with volume confirmation (>1.5x 20-period average) and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Enters long on upper band breakout in trending regime or lower band bounce in choppy regime; short on lower band breakout in trending regime or upper band bounce in choppy regime. Uses 1d HTF EMA(50) for trend alignment. Discrete position sizing (0.25) to limit fee drag. Target: 20-50 trades/year to work in both bull and bear markets by combining breakout and mean-reversion logic based on volatility regime.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    wma2 = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - close.shift(1))), np.abs(low - close.shift(1))))
    sum_atr = atr.rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

name = "4h_donchian_breakout_volume_chop_regime_v4"
timeframe = "4h"
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
    
    # 1d HTF trend filter: 50-period EMA on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(chop[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filters
        choppy_market = chop[i] > 61.8  # Mean reversion regime
        trending_market = chop[i] < 38.2  # Trend following regime
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower band OR stops losing momentum
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band OR stops losing momentum
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation
            if volume_confirmed:
                # Bullish 1d trend alignment
                bullish_htf = close[i] > ema_50_1d_aligned[i]
                bearish_htf = close[i] < ema_50_1d_aligned[i]
                
                # In choppy market: mean reversion at Donchian bands
                if choppy_market:
                    # Long: price near lower band with bullish HTF alignment
                    if close[i] <= lowest_low[i] * 1.001 and bullish_htf:  # Allow tiny buffer for precision
                        position = 1
                        signals[i] = 0.25
                    # Short: price near upper band with bearish HTF alignment
                    elif close[i] >= highest_high[i] * 0.999 and bearish_htf:
                        position = -1
                        signals[i] = -0.25
                # In trending market: breakout continuation
                elif trending_market:
                    # Long: breakout above upper band with bullish HTF alignment
                    if close[i] > highest_high[i] and bullish_htf:
                        position = 1
                        signals[i] = 0.25
                    # Short: breakout below lower band with bearish HTF alignment
                    elif close[i] < lowest_low[i] and bearish_htf:
                        position = -1
                        signals[i] = -0.25
    
    return signals