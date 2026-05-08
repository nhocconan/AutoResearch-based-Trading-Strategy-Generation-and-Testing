#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Chandelier Exit trailing stop with 1d EMA trend filter and volume confirmation
# Chandelier Exit uses ATR to set dynamic stop levels that trail price in trending markets.
# Long position: price above 1d EMA(50) + long Chandelier Exit (high - ATR*multiplier) rising
# Short position: price below 1d EMA(50) + short Chandelier Exit (low + ATR*multiplier) falling
# Volume spike confirms momentum. Designed to capture trends while limiting drawdowns.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_ChandelierExit_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(22) for Chandelier Exit
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=22, min_periods=22).mean().values
    
    # Chandelier Exit components (2.5 multiplier)
    mult = 2.5
    long_ce = np.maximum.accumulate(high) - mult * atr  # for longs: trail below highs
    short_ce = np.minimum.accumulate(low) + mult * atr   # for shorts: trail above lows
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(long_ce[i]) or 
            np.isnan(short_ce[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price above 1d EMA + Chandelier Exit rising + volume spike
            if (close[i] > ema50_1d_val and 
                long_ce[i] > long_ce[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price below 1d EMA + Chandelier Exit falling + volume spike
            elif (close[i] < ema50_1d_val and 
                  short_ce[i] < short_ce[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Chandelier Exit OR trend reverses
            if (close[i] <= long_ce[i] or close[i] < ema50_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Chandelier Exit OR trend reverses
            if (close[i] >= short_ce[i] or close[i] > ema50_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals