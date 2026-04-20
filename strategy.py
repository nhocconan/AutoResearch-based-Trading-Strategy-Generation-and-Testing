# 1. State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 6h session momentum (UTC 08:00-16:00) + 1d volume confirmation + 1d ATR stop
# Session filter captures institutional activity hours, volume confirms institutional participation,
# ATR stop adapts to volatility. Works in bull/bear by only trading during active sessions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d ATR (14-period) for stop loss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Pre-compute session hours (UTC 8-16)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 16)
    
    # 6h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if NaN in critical values
        if np.isnan(vol_ma_1d_6h[i]) or np.isnan(atr_1d_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: session hour + volume > 1.5x 1d average + price > open (bullish candle)
            if vol > 1.5 * vol_ma_1d_6h[i] and price > prices['open'].iloc[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: session hour + volume > 1.5x 1d average + price < open (bearish candle)
            elif vol > 1.5 * vol_ma_1d_6h[i] and price < prices['open'].iloc[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: session end OR ATR stop hit (2*ATR)
            if not in_session[i] or price < entry_price - 2.0 * atr_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: session end OR ATR stop hit (2*ATR)
            if not in_session[i] or price > entry_price + 2.0 * atr_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_SessionMomentum_Volume_ATRStop"
timeframe = "6h"
leverage = 1.0