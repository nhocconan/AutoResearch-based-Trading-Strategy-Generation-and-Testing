#!/usr/bin/env python3
"""
1d_KAMA_Regime_Volume_Breakout_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with choppiness regime filter and volume confirmation to avoid whipsaws.
Enter long when price > KAMA + chop regime indicates trending + volume > 1.5x average.
Enter short when price < KAMA + chop regime indicates trending + volume > 1.5x average.
Exit when price crosses back below/above KAMA or chop regime shifts to ranging.
This adaptive approach works in both bull (KAMA catches trends) and bear (KAMA adapts quickly to downtrends)
markets while avoiding false signals in ranging conditions via chop filter.
Target: 20-80 trades over 4 years (5-20/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (more stable than 1d for regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need sufficient data for weekly calculations
        return np.zeros(n)
    
    # Calculate weekly EMA20 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA on daily close
    # KAMA parameters: ER length=10, fast=2, slow=30
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate Choppiness Index on daily data (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    high_roll = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_roll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (high_roll - low_roll)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((high_roll - low_roll) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), ATR (14), volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(chop[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above weekly EMA20)
        htf_1w_bullish = close[i] > ema_20_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_20_1w_aligned[i]
        
        # Regime filter: chop < 38.2 = trending (good for breakout/trend following)
        # chop > 61.8 = ranging (avoid trend signals)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price > KAMA + trending regime + volume confirmation + 1w uptrend
            long_setup = (close[i] > kama[i]) and trending_regime and volume_confirm and htf_1w_bullish
            
            # Short setup: price < KAMA + trending regime + volume confirmation + 1w downtrend
            short_setup = (close[i] < kama[i]) and trending_regime and volume_confirm and htf_1w_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price < KAMA OR chop > 50 (ranging) OR 1w trend turns bearish
            if (close[i] < kama[i]) or (chop[i] > 50) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price > KAMA OR chop > 50 (ranging) OR 1w trend turns bullish
            if (close[i] > kama[i]) or (chop[i] > 50) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Regime_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0