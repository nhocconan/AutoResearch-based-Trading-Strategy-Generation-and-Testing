#!/usr/bin/env python3
"""
1h 4h/1d Trend + Pullback Strategy
Hypothesis: In both bull and bear markets, trends persist on higher timeframes (4h/1d).
We use 4h EMA50 and 1d EMA100 to establish trend direction, then enter on 1h pullbacks
to the 21 EMA with volume confirmation. This captures trend continuation moves while
avoiding counter-trend trades. Target: 15-37 trades/year by using tight entry conditions
(close to EMA21 + volume spike + strong trend alignment).
"""

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
    
    # Get 4h and 1d data for trend filters (once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 100:
        return np.zeros(n)
    
    # 4h EMA50 for intermediate trend
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA100 for long-term trend
    ema100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 1h EMA21 for entry timing
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(ema21[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema21_val = ema21[i]
        atr_val = atr[i]
        vol_ok = vol_filter[i]
        trend_4h = ema50_4h_aligned[i]
        trend_1d = ema100_1d_aligned[i]
        
        # Distance from EMA21 in ATR units
        dist_to_ema21 = abs(price - ema21_val) / atr_val
        
        if position == 0:
            # Long: price near EMA21 pullback in uptrend (both timeframes aligned)
            if (price > ema21_val and 
                dist_to_ema21 < 0.5 and  # Within 0.5 ATR of EMA21
                vol_ok and 
                price > trend_4h and 
                price > trend_1d):
                signals[i] = 0.20
                position = 1
            # Short: price near EMA21 pullback in downtrend (both timeframes aligned)
            elif (price < ema21_val and 
                  dist_to_ema21 < 0.5 and  # Within 0.5 ATR of EMA21
                  vol_ok and 
                  price < trend_4h and 
                  price < trend_1d):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit if trend breaks or price moves too far from EMA21
            if (price < trend_4h or 
                price < trend_1d or 
                dist_to_ema21 > 1.5):  # Too far from EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit if trend breaks or price moves too far from EMA21
            if (price > trend_4h or 
                price > trend_1d or 
                dist_to_ema21 > 1.5):  # Too far from EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Trend_Pullback_EMA21"
timeframe = "1h"
leverage = 1.0