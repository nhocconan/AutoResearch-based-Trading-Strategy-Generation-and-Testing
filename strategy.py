#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1-day ATR-based volatility breakout with 1-week trend filter and volume confirmation.
In uptrend (price > 200-period EMA), buy when price breaks above open + ATR(14); in downtrend (price < 200-period EMA), sell when price breaks below open - ATR(14).
ATR breakouts capture volatility expansion; EMA200 filters for long-term trend alignment; volume confirms breakout strength.
Works in bull markets (buy breakouts) and bear markets (sell breakdowns). Target: 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate ATR(14) from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = tr1[0]  # First period
    tr3[0] = tr1[0]  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Open prices from 1d for breakout levels
    open_1d = df_1d['open'].values
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    
    # 4h volume confirmation (volume spike > 2.0x 50-period average)
    vol_ma_50 = pd.Series(prices['volume'].values).rolling(window=50, min_periods=50).mean().values
    vol_ratio = prices['volume'].values / vol_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(open_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_200_aligned[i]
        atr_val = atr_14_aligned[i]
        open_val = open_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 2.0  # Volume spike filter
        
        if position == 0:
            # Enter long: price breaks above open + ATR + uptrend (price > EMA200) + volume spike
            if (price_close > open_val + atr_val and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below open - ATR + downtrend (price < EMA200) + volume spike
            elif (price_close < open_val - atr_val and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA200 in opposite direction)
            if position == 1 and price_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ATRBreakout_1wEMA200_Trend_Volume"
timeframe = "4h"
leverage = 1.0