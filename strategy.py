#!/usr/bin/env python3
"""
4h_Keltner_Breakout_1dTrend_Volume
Hypothesis: Price breakout beyond Keltner Channel (2*ATR) with 1d EMA50 trend filter and volume confirmation. Works in bull markets by capturing breakouts and in bear markets by following higher timeframe trend. Target 20-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === ATR(10) for Keltner Channel ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === EMA(20) for Keltner base ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Keltner Channel bounds ===
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(kc_upper[i]) or
            np.isnan(kc_lower[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_open = prices['open'].values[i]
        trend_1d = ema_50_1d_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Close above Keltner Upper + volume spike > 1.5 + price above 1d EMA50
            if (price_close > kc_upper[i] and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Close below Keltner Lower + volume spike > 1.5 + price below 1d EMA50
            elif (price_close < kc_lower[i] and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses back below/above EMA(20)
            if position == 1 and price_close < ema_20[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Keltner_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0