#!/usr/bin/env python3
name = "1h_OrderBlock_Momentum_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0

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
    
    # === 4h Trend Filter (HTF) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h Order Block Detection (LTF) ===
    # Bullish OB: strong bullish candle after consolidation (low volume then high volume)
    # Bearish OB: strong bearish candle after consolidation
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Body size and wick ratio
    body_size = np.abs(close - open_)
    total_range = high - low
    body_ratio = np.divide(body_size, total_range, out=np.zeros_like(body_size), where=total_range!=0)
    
    # Strong candle: body > 60% of range
    strong_candle = body_ratio > 0.6
    
    # Bullish/bearish candle
    bullish_candle = close > open_
    bearish_candle = close < open_
    
    # Volume surge: current volume > 1.5x average
    vol_surge = volume > (vol_ma_20 * 1.5)
    
    # Order block conditions
    bullish_ob = bullish_candle & strong_candle & vol_surge
    bearish_ob = bearish_candle & strong_candle & vol_surge
    
    # === Entry Conditions ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish OB in 4h uptrend
            if (bullish_ob[i] and 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and  # 4h uptrend
                close[i] > close[i-1]):  # additional confirmation
                signals[i] = 0.20
                position = 1
            # Short: bearish OB in 4h downtrend
            elif (bearish_ob[i] and 
                  ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and  # 4h downtrend
                  close[i] < close[i-1]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend reversal or opposite OB
            if (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] or  # trend reversal
                bearish_ob[i]):  # opposite signal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h trend reversal or opposite OB
            if (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] or  # trend reversal
                bullish_ob[i]):  # opposite signal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Order Block momentum with 4h trend filter
# - Order Blocks represent institutional footprint where price is likely to return
# - Bullish OB: strong bullish candle with volume surge after consolidation
# - Bearish OB: strong bearish candle with volume surge after consolidation
# - 4h EMA50 trend filter ensures we only trade in direction of higher timeframe trend
# - Works in bull markets (buy bullish OBs in uptrend) and bear markets (sell bearish OBs in downtrend)
# - Volume surge filter reduces false signals
# - Position size 0.20 targets ~20-40 trades/year to minimize fee drag
# - Uses 4h for trend direction (structure) and 1h for precise entry timing
# - Avoids overtrading by requiring multiple confluence factors (OB + trend + volume)