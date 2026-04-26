#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dEMA50_Trend_VolumeChop_v1
Hypothesis: On 12h timeframe, Donchian(20) breakouts with 1d EMA50 trend filter, volume confirmation (1.5x 20-bar MA), and choppiness regime (CHOP<50) produce high-quality trades with low frequency. The chop filter avoids whipsaws in ranging markets, improving performance in both bull and bear regimes. Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h choppiness index: CHOP > 61.8 = ranging (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.concatenate([[np.nan], close[:-1]])))
    tr2 = np.maximum(tr1, np.absolute(low - np.concatenate([[np.nan], close[:-1]])))
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid div by zero
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian/volume MA, 14*2 for chop)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Choppiness regime: only take breakouts when CHOP < 50 (less choppy/more trending)
        regime_ok = chop[i] < 50.0
        
        # Donchian breakout conditions
        breakout_high = close[i] > donchian_high[i]
        breakout_low = close[i] < donchian_low[i]
        
        # Long logic: breakout above Donchian high in uptrend with volume and good regime
        if uptrend and volume_spike and breakout_high and regime_ok:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below Donchian low in downtrend with volume and good regime
        elif downtrend and volume_spike and breakout_low and regime_ok:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend OR regime becomes too choppy
        elif position == 1 and (not uptrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeChop_v1"
timeframe = "12h"
leverage = 1.0