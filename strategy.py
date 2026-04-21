#!/usr/bin/env python3
"""
4h_KeltnerBreakout_1dTrend_Volume
Hypothesis: Keltner Channel breakout (ATR-based) with 1d EMA50 trend filter and volume confirmation. 
Designed to catch breakouts in trending markets while avoiding false signals in ranging markets. 
Works in bull/bear by following higher timeframe trend. Target 20-40 trades/year on 4h.
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
    
    # === Keltner Channel (ATR-based) on 4h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA(20) of close
    ema_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR(10)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Upper and Lower Bands
    keltner_upper = ema_mid + 2.0 * atr
    keltner_lower = ema_mid - 2.0 * atr
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_mid[i]) or
            np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1d = ema_50_1d_aligned[i]
        upper = keltner_upper[i]
        lower = keltner_lower[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Close breaks above upper band + volume spike > 1.5 + price above 1d EMA50
            if (price_close > upper and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower band + volume spike > 1.5 + price below 1d EMA50
            elif (price_close < lower and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to middle band
            if position == 1 and price_close < ema_mid[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_KeltnerBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0