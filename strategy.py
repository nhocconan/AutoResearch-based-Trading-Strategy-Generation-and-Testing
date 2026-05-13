#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (HMA34) and volume confirmation (2.5x MA20).
# Enters long when price breaks above 12h Donchian high with 1d bullish trend and volume > 2.5x MA20.
# Enters short when price breaks below 12h Donchian low with 1d bearish trend and volume > 2.5x MA20.
# Exits when price crosses the 12h EMA34 (mean reversion).
# Uses discrete position sizing (0.30) to limit fee churn and manage drawdown.
# Target: ~30-50 trades/year on 12h timeframe by requiring strict confluence.
# Works in both bull and bear markets: 1d trend filter ensures alignment with higher timeframe direction,
# while Donchian breakouts capture strong momentum moves and volume confirmation reduces false signals.

name = "12h_Donchian20_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 12h data for Donchian channels (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels: upper, lower (based on previous 12h bar)
    lookback = 20
    donchian_high = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for trend filter (HMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate HMA(34) on 1d close
    half_len = 34 // 2
    sqrt_len = int(np.sqrt(34))
    wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_34 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_34_aligned = align_htf_to_ltf(prices, df_1d, hma_34)
    
    # Volume filter: current volume > 2.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.5)
    
    # 12h EMA34 for exit condition
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(hma_34_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema34_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with 1d bullish trend and volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > hma_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian low with 1d bearish trend and volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < hma_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 12h EMA34 (mean reversion)
            if close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above 12h EMA34 (mean reversion)
            if close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals