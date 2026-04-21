#!/usr/bin/env python3
"""
6h_ParabolicSAR_RangeBreakout_1dTrend
Hypothesis: Use 1d EMA50 for trend direction, Parabolic SAR (0.02, 0.2) for breakout detection, and volume confirmation. 
Parabolic SAR performs well in trending markets but whipsaws in ranges; combining with 1d trend filter avoids counter-trend trades. 
Volume > 1.3x average confirms breakout strength. Works in bull/bear markets by following higher timeframe trend. Target 20-40 trades/year on 6h.
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
    
    # === Parabolic SAR (0.02, 0.2) on 6h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Initialize arrays
    psar = np.zeros(n)
    psar[0] = low[0]
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = high[0] if trend == 1 else low[0]  # extreme point
    
    for i in range(1, n):
        if trend == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR does not exceed previous two lows
            psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            # Trend reversal
            if low[i] < psar[i]:
                trend = -1
                psar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                # Continue uptrend
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR does not fall below previous two highs
            psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
            # Trend reversal
            if high[i] > psar[i]:
                trend = 1
                psar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                # Continue downtrend
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(psar[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1d = ema_50_1d_aligned[i]
        psar_val = psar[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price > PSAR + volume spike > 1.3 + price above 1d EMA50 (uptrend)
            if (price_close > psar_val and 
                vol_spike > 1.3 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: price < PSAR + volume spike > 1.3 + price below 1d EMA50 (downtrend)
            elif (price_close < psar_val and 
                  vol_spike > 1.3 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses PSAR in opposite direction
            if position == 1 and price_close < psar_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > psar_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ParabolicSAR_RangeBreakout_1dTrend"
timeframe = "6h"
leverage = 1.0