# [EXPERIMENT #100322] 12h Donchian(20) breakout + 1d trend + volume confirmation + ATR stoploss
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# Works in bull/bear: breakout + 1d trend filter prevents whipsaws in chop, volume avoids fakeouts
# Uses 12h timeframe as requested, with 1d HTF for trend filter
# Expected: ~30-50 trades total (well under 200 hard max)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # ATR(20) for stoploss and breakout sensitivity
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 20-period EMA on daily close for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume filter: volume > 1.3x 30-period average (looser than 2.0x to allow more signals)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above 20-period high AND 1d uptrend AND volume
        if (close[i] > highest_high[i-1] and 
            close[i] > ema20_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below 20-period low AND 1d downtrend AND volume
        elif (close[i] < lowest_low[i-1] and 
              close[i] < ema20_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0