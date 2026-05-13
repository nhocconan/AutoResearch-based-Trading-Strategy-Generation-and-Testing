#!/usr/bin/env python3
"""
12h_1d_AdaptiveKeltnerBreakout
Hypothesis: Keltner Channel breakouts with ATR-based filtering and 1d trend alignment work in both bull and bear markets.
Breakouts above upper channel with 1d uptrend and volume confirmation signal long.
Breakdowns below lower channel with 1d downtrend and volume confirmation signal short.
ATR filter ensures volatility is sufficient to avoid chop. Uses Keltner (EMA-based) for smoother bands vs Bollinger.
Target: 12-37 trades/year per symbol.
"""

name = "12h_1d_AdaptiveKeltnerBreakout"
timeframe = "12h"
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
    
    # Get 1d data for trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = np.zeros_like(close_1d)
    for i in range(14, len(tr)):
        if np.isnan(tr[i]):
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.nanmean(tr[i-13:i+1])
    atr_14 = np.where(np.isnan(atr_14), 0, atr_14)
    
    # Keltner Channel on 12h: EMA(20) ± ATR(10)*2
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = np.zeros(n)
    for i in range(10, n):
        tr_l = np.maximum(high[i] - low[i], np.abs(high[i] - close[i-1]))
        tr_l = np.maximum(np.abs(low[i] - close[i-1]), tr_l)
        atr_10[i] = np.mean(np.abs(np.concatenate([[np.nan], [tr_l]]))[i-9:i+1]) if not np.isnan(tr_l) else 0
    atr_10 = np.where(np.isnan(atr_10), 0, atr_10)
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    
    # 1d trend: EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend and ATR to 12h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values for current bar
        upper = kc_upper[i]
        lower = kc_lower[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        atr_val = atr_14_aligned[i]
        
        # Skip if ATR too low (avoid chop)
        if atr_val < 0.005 * close[i]:  # less than 0.5% of price
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: price breaks above upper KC, 1d uptrend, volume confirmation
            if close[i] > upper and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower KC, 1d downtrend, volume confirmation
            elif close[i] < lower and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below lower KC or 1d trend turns down
            if close[i] < lower or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above upper KC or 1d trend turns up
            if close[i] > upper or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals