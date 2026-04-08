#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and 12h volume confirmation.
# In bull markets: breakout above upper band in uptrend (price > 1d EMA50).
# In bear markets: breakout below lower band in downtrend (price < 1d EMA50).
# Volume surge filters false breakouts. ATR-based stop limits drawdown.
# Target: 15-35 trades/year on 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price below lower Donchian band OR stoploss hit
            if close[i] < low_min[i] or close[i] < ema50_1d_aligned[i] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above upper Donchian band OR stoploss hit
            if close[i] > high_max[i] or close[i] > ema50_1d_aligned[i] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band, uptrend (price > 1d EMA50), volume surge
            if close[i] > high_max[i] and close[i] > ema50_1d_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band, downtrend (price < 1d EMA50), volume surge
            elif close[i] < low_min[i] and close[i] < ema50_1d_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals