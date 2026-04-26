#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRVolumeRegime
Hypothesis: 4-hour Donchian channel breakout with ATR-based volume confirmation and chop regime filter.
Enters long when price breaks above upper band with expanding volume and low chop (trending market).
Enters short when price breaks below lower band with expanding volume and low chop.
Uses ATR trailing stop for risk management. Discrete position sizing (0.0, ±0.30) minimizes fee churn.
Designed to work in both bull and bear markets by adapting to regime via chop filter.
Target: 20-50 trades/year per symbol (<200 total over 4 years).
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
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR (14-period) for volatility and stoploss
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_expanding = volume > (1.5 * avg_volume)
    
    # Choppiness regime filter (14-period) - from 1d timeframe to avoid look-ahead
    df_1d = get_htf_data(prices, '1d')
    chop_period = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Sum of true range over chop_period
    sum_tr = pd.Series(tr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_1d = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Choppiness index: 100 * log10(sum_tr / (highest_high_1d - lowest_low_1d)) / log10(chop_period)
    denominator = highest_high_1d - lowest_low_1d
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(sum_tr / denominator) / np.log10(chop_period)
    
    # Align chop to 4h timeframe (wait for completed 1d bar + extra delay for chop confirmation)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Chop regime: < 38.2 = trending (favor breakouts), > 61.8 = ranging (favor mean reversion)
    # We want trending regime for breakouts
    trending_regime = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need 20-period lookback + 14-period ATR + chop calculation)
    start_idx = max(lookback, atr_period, chop_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_expanding[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(trending_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above upper Donchian + volume expanding + trending regime
        if close[i] > highest_high[i] and volume_expanding[i] and trending_regime[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below lower Donchian + volume expanding + trending regime
        elif close[i] < lowest_low[i] and volume_expanding[i] and trending_regime[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: ATR trailing stop
        elif position == 1 and close[i] < (highest_high[i] - 2.0 * atr[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (lowest_low[i] + 2.0 * atr[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_ATRVolumeRegime"
timeframe = "4h"
leverage = 1.0