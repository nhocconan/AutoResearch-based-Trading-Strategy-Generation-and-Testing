#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR(14) volatility filter.
# Long when price breaks above upper Donchian channel AND 1d EMA34 uptrend AND ATR(14) < median ATR(50).
# Short when price breaks below lower Donchian channel AND 1d EMA34 downtrend AND ATR(14) < median ATR(50).
# Uses volatility contraction breakout pattern: low volatility precedes expansion breakouts.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_Breakout_1dEMA34_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate median ATR(50) for volatility regime filter
    median_atr_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Donchian, EMA, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(median_atr_50[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volatility filter: current ATR < median ATR (low volatility environment)
        if median_atr_50[i] <= 0:
            vol_filter = False
        else:
            vol_filter = atr_14[i] < median_atr_50[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND uptrend AND low volatility
            if curr_high > highest_20[i] and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND downtrend AND low volatility
            elif curr_low < lowest_20[i] and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls back below upper Donchian OR trend turns down
            if curr_close < highest_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises back above lower Donchian OR trend turns up
            if curr_close > lowest_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals