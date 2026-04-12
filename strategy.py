#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_donchian_volume_chop_v1
# Uses daily Donchian channel breakout with volume confirmation and chop filter.
# Buys when price breaks above daily Donchian(20) high with volume > 1.5x average and chop < 61.8 (trending).
# Shorts when price breaks below daily Donchian(20) low with volume > 1.5x average and chop < 61.8.
# Uses 4h timeframe for entries to reduce trade frequency vs daily while capturing daily breakouts.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (daily levels update only after daily bar closes)
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Chop filter: calculate Choppy Index (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_sum = np.convolve(tr, np.ones(14), 'valid')
    atr_sum = np.concatenate([np.full(13, np.nan), atr_sum])
    
    hh = np.maximum.accumulate(high)
    ll = np.minimum.accumulate(low)
    hhll = hh - ll
    
    chop = 100 * np.log10(atr_sum / hhll) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])
    
    # Chop < 61.8 indicates trending market (good for breakouts)
    chop_filter = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or np.isnan(chop_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and chop filters
        if not (vol_confirm[i] and chop_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above daily Donchian high with volume
        if close[i] > donchian_high_4h[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below daily Donchian low with volume
        elif close[i] < donchian_low_4h[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < donchian_low_4h[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > donchian_high_4h[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals