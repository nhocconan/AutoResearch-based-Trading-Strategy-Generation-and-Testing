#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 12h Donchian channel breakout with volume confirmation (>1.5x 20-period average) and 1d HTF trend filter (price > 50-period EMA). Enters long when price breaks above upper Donchian(20) with volume confirmation and bullish 1d trend; short when price breaks below lower Donchian(20) with volume confirmation and bearish 1d trend. Exits on opposite Donchian level touch. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 12-37 trades/year) to work in both bull and bear markets by following institutional volume-driven breakouts in alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channel (20-period) on 12h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Previous day's OHLC for 1d HTF data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d HTF trend filter: 50-period EMA on 1d timeframe
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks lower Donchian level
            if close[i] <= lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks upper Donchian level
            if close[i] >= highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and 1d trend alignment
            if volume_confirmed:
                # Bullish 1d trend: price above 50-period EMA
                bullish_trend = close[i] > ema_50_1d_aligned[i]
                # Bearish 1d trend: price below 50-period EMA
                bearish_trend = close[i] < ema_50_1d_aligned[i]
                
                # Long: price breaks above upper Donchian with volume and bullish 1d trend
                if close[i] > highest_high[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian with volume and bearish 1d trend
                elif close[i] < lowest_low[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals