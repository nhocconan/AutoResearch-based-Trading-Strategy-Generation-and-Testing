#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: Trade Donchian(20) breakouts on 12h with 1d trend filter and volume confirmation. 
Breakouts capture strong momentum moves. The 1d trend filter ensures we only trade in the direction 
of the higher timeframe trend, reducing false signals during counter-trend moves. Volume confirmation 
adds conviction. Works in bull markets by catching breakouts and in bear markets by avoiding 
counter-trend breakouts via the 1d trend filter. Targets 15-30 trades/year via strict breakout 
conditions + trend filter + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False).values
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian(20) channels on 12h
    lookback = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, n):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, 34)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + 1d uptrend + volume
            if close[i] > upper[i] and close[i] > ema_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + 1d downtrend + volume
            elif close[i] < lower[i] and close[i] < ema_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or 1d trend turns down
            if close[i] < lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or 1d trend turns up
            if close[i] > upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0