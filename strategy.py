#!/usr/bin/env python3
# 12h_donchian_breakout_volume_atr_v1
# Hypothesis: 12h Donchian(20) breakouts with volume confirmation and ATR-based trend filter.
# Long when price breaks above 20-bar high with volume > 1.5x average and ATR(14) rising.
# Short when price breaks below 20-bar low with volume > 1.5x average and ATR(14) rising.
# Exits on opposite Donchian breakout or ATR mean reversion (ATR < ATR_ma).
# Works in bull/bear: ATR filter ensures volatility expansion, volume confirms validity.
# Target: 12-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for ATR trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w ATR(14)
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 1w ATR(14) moving average for trend filter
    atr_ma_1w = pd.Series(atr_1w_aligned).rolling(window=20, min_periods=20).mean().values
    
    # 12h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(atr_ma_1w[i]) or 
            np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 12h Donchian low OR ATR mean reverts (ATR < ATR_ma)
            if low[i] < low_ma[i] or atr_1w_aligned[i] < atr_ma_1w[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 12h Donchian high OR ATR mean reverts (ATR < ATR_ma)
            if high[i] > high_ma[i] or atr_1w_aligned[i] < atr_ma_1w[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long entry: price breaks above 12h Donchian high with rising ATR
                if high[i] > high_ma[i] and atr_1w_aligned[i] > atr_ma_1w[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below 12h Donchian low with rising ATR
                elif low[i] < low_ma[i] and atr_1w_aligned[i] > atr_ma_1w[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals