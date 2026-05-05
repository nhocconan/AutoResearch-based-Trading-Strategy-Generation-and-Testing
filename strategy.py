#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR(14) volatility filter
# Long when price breaks above Donchian upper(20) AND 12h close > 12h EMA50 AND ATR(14) < 0.03 * close
# Short when price breaks below Donchian lower(20) AND 12h close < 12h EMA50 AND ATR(14) < 0.03 * close
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-30 trades/year per symbol.
# Donchian provides structure; EMA50 filters trend; ATR filter avoids high volatility chop.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# 4h timeframe balances trade frequency and trend capture.

name = "4h_Donchian20_12hEMA50_ATR_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Calculate Donchian(20) on 4h data
    # Upper band: highest high of last 20 bars
    # Lower band: lowest low of last 20 bars
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any value is NaN
        if (np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper AND 12h uptrend AND low volatility
            if (close[i] > highest_high[i] and 
                uptrend_12h_aligned[i] > 0.5 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower AND 12h downtrend AND low volatility
            elif (close[i] < lowest_low[i] and 
                  downtrend_12h_aligned[i] > 0.5 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian lower OR 12h trend changes to downtrend
            if (close[i] < lowest_low[i] or 
                downtrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian upper OR 12h trend changes to uptrend
            if (close[i] > highest_high[i] or 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals