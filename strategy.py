#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATRTrend_VolumeConfirm
Hypothesis: 4h Donchian channel breakout with 1-day ATR-based trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) with bullish 1d ATR trend and volume spike.
Short when price breaks below lower Donchian(20) with bearish 1d ATR trend and volume spike.
ATR trend filter adapts to volatility regimes, reducing false breakouts in choppy markets.
Volume confirmation ensures institutional participation. Target 30-60 trades/year.
Works in bull markets via breakout continuation and bear markets via volatility expansion breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ATR-based trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily timeframe
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # ATR trend: rising ATR = increasing volatility (bullish for breakouts)
    atr_ma_10 = pd.Series(atr_14_1d).rolling(window=10, min_periods=10).mean().values
    atr_trend_bullish = atr_14_1d > atr_ma_10  # ATR above its MA = expanding volatility
    atr_trend_bearish = atr_14_1d < atr_ma_10  # ATR below its MA = contracting volatility
    
    # Align ATR trend to 4h timeframe
    atr_trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_bullish.astype(float))
    atr_trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_bearish.astype(float))
    
    # Donchian channel (20-period) on 4h timeframe
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20), ATR(14) with MA(10), volume MA
    start_idx = max(20, 14+10, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or 
            np.isnan(atr_trend_bullish_aligned[i]) or 
            np.isnan(atr_trend_bearish_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian + bullish ATR trend + volume spike
            long_setup = (close[i] > high_ma_20[i]) and atr_trend_bullish_aligned[i] and volume_spike[i]
            # Short: break below lower Donchian + bearish ATR trend + volume spike
            short_setup = (close[i] < low_ma_20[i]) and atr_trend_bearish_aligned[i] and volume_spike[i]
            
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
            # Exit: price closes below lower Donchian OR ATR trend turns bearish
            if (close[i] < low_ma_20[i]) or (~atr_trend_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above upper Donchian OR ATR trend turns bullish
            if (close[i] > high_ma_20[i]) or atr_trend_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dATRTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0