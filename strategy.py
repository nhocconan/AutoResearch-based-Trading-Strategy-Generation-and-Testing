#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: 12-hour Donchian breakout with 1-day trend filter and volume confirmation.
# Enters long when price breaks above 20-period Donchian upper band in daily uptrend with volume confirmation.
# Enters short when price breaks below 20-period Donchian lower band in daily downtrend with volume confirmation.
# Uses ATR-based stop loss and position sizing of 0.25 to manage risk.
# Designed to capture medium-term trends while avoiding false breakouts in low volatility.
# Target: 12-30 trades per year for low fee drag on 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (1d EMA200) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 12h indicators
    # Donchian channels (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 60  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema200_1d_aligned[i]
        daily_downtrend = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: close below Donchian lower band or volatility contraction
            if close[i] < low_20[i] or atr[i] < atr[i-1] * 0.7:  # Volatility dropping
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: close above Donchian upper band or volatility contraction
            if close[i] > high_20[i] or atr[i] < atr[i-1] * 0.7:  # Volatility dropping
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Breakout conditions
            if volume_ok:
                # Long breakout: price crosses above upper Donchian band in uptrend
                if daily_uptrend and close[i] > high_20[i] and close[i-1] <= high_20[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price crosses below lower Donchian band in downtrend
                elif daily_downtrend and close[i] < low_20[i] and close[i-1] >= low_20[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals