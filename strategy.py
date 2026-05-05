#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and ATR(14) volatility filter
# Long when price breaks above 6h Donchian high(20) AND 1w close > 1w EMA50 AND ATR(14) < 0.03 * close
# Short when price breaks below 6h Donchian low(20) AND 1w close < 1w EMA50 AND ATR(14) < 0.03 * close
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Donchian provides structure; 1w EMA50 filters primary trend; ATR filter avoids high volatility chop.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# 6h timeframe balances trade frequency and responsiveness while minimizing fee drag.

name = "6h_Donchian20_1wEMA50_ATR_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels based on previous 6h bar
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian(20): 20-period high and low
    high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Shift to use previous bar's levels (breakout of previous bar's Donchian)
    high_20 = np.roll(high_20, 1)
    low_20 = np.roll(low_20, 1)
    high_20[0] = np.nan  # First value invalid after roll
    low_20[0] = np.nan
    
    # Align Donchian levels to prices timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_6h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_6h, low_20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 6h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Volatility filter: ATR < 3% of price (avoid high volatility chop)
    vol_filter = atr_14 < (0.03 * close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian high(20) AND 1w uptrend AND low volatility
            if (close[i] > high_20_aligned[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian low(20) AND 1w downtrend AND low volatility
            elif (close[i] < low_20_aligned[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian low(20) OR 1w trend changes to downtrend
            if (close[i] < low_20_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian high(20) OR 1w trend changes to uptrend
            if (close[i] > high_20_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals