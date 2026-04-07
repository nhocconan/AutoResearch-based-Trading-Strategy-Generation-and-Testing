#!/usr/bin/env python3
"""
4h_adx_trend_v1
Hypothesis: On 4h timeframe, use ADX > 25 for trending market filter, with EMA crossover (12/26) for entry signals.
Enter long when EMA12 > EMA26 and ADX > 25, short when EMA12 < EMA26 and ADX > 25.
Exit when ADX drops below 20 or EMA crossover reverses. Uses ADX to avoid whipsaw in ranging markets.
Targets 20-50 trades/year to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adx_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA crossover (fast=12, slow=26)
    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26 = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after ADX warmup
        # Skip if required data not available
        if (np.isnan(ema12[i]) or np.isnan(ema26[i]) or 
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend strength filter
        trending = adx[i] > 25
        ranging = adx[i] < 20
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on trend weakening (ADX < 20)
            if adx[i] < 20:
                exit_long = True
            # Exit on EMA crossover reversal
            elif ema12[i] < ema26[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on trend weakening (ADX < 20)
            if adx[i] < 20:
                exit_short = True
            # Exit on EMA crossover reversal
            elif ema12[i] > ema26[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in trending markets (ADX > 25)
            if trending:
                # Long entry: EMA12 > EMA26
                if ema12[i] > ema26[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: EMA12 < EMA26
                elif ema12[i] < ema26[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals