#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (HMA21) and volume confirmation (2.0x MA20).
# Enters long when price breaks above Donchian high with 1w bullish trend and volume > 2.0x MA20.
# Enters short when price breaks below Donchian low with 1w bearish trend and volume > 2.0x MA20.
# Exits when price crosses the 1d EMA20 (mean reversion).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~10-25/year) by requiring strict confluence.
# Works in both bull and bear markets: 1w trend filter ensures alignment with higher timeframe direction,
# while Donchian breakouts capture strong momentum moves and volume confirmation reduces false signals.

name = "1d_Donchian20_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels: upper, lower (based on previous 1d bar)
    lookback = 20
    donchian_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1w data for trend filter (HMA21)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate HMA(21) on 1w close
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # 1d EMA20 for exit condition
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with 1w bullish trend and volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > hma_21_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with 1w bearish trend and volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < hma_21_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1d EMA20 (mean reversion)
            if close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1d EMA20 (mean reversion)
            if close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals