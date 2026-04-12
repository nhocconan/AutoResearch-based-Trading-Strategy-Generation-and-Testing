#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 200-period EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily 200-period EMA
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr6 = np.full(n, np.nan)
    for i in range(5, n):
        atr6[i] = np.nanmean(tr[i-5:i+1])
    
    # Calculate 20-period ATR EMA for volatility regime
    atr_ema20 = np.full(n, np.nan)
    atr_series = pd.Series(atr6)
    atr_ema20_values = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ema20[:] = atr_ema20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr6[i]) or 
            np.isnan(atr_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR6 > 1.2x 20-period ATR EMA (high volatility regime)
        vol_filter = atr6[i] > atr_ema20[i] * 1.2
        
        # Trend filter: price above/below daily 200 EMA
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        price_below_ema200 = close[i] < ema200_1d_aligned[i]
        
        # Entry conditions: breakout in direction of trend with volatility expansion
        long_breakout = close[i] > high[i-1]  # break above previous high
        short_breakout = close[i] < low[i-1]  # break below previous low
        
        long_entry = long_breakout and price_above_ema200 and vol_filter
        short_entry = short_breakout and price_below_ema200 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema200_1d_aligned[i]) or (atr6[i] < atr_ema20[i] * 0.8)
        short_exit = (close[i] > ema200_1d_aligned[i]) or (atr6[i] < atr_ema20[i] * 0.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1d_ema200_breakout_vol_filter_v1"
timeframe = "6h"
leverage = 1.0