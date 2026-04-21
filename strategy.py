#!/usr/bin/env python3
"""
6h_12h_1d_Donchian20_Breakout_VolumeATRFilter_v1
Hypothesis: Donchian(20) breakout on 6h timeframe with 12h trend filter (EMA34) and 1d volume spike + ATR filter.
Works in bull/bear: Breakouts capture momentum in both directions. Volume confirms conviction, ATR filters low-volatility false breakouts.
Target: 12-30 trades/year per symbol (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data once for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Load 1d data once for volume and ATR filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d volume MA(20) for volume spike filter
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Donchian(20) on 6h: lookback 20 periods (excluding current)
        if i >= 20:
            lookback_start = i - 20
            lookback_end = i  # exclusive
            high_20 = prices['high'].iloc[lookback_start:lookback_end].max()
            low_20 = prices['low'].iloc[lookback_start:lookback_end].min()
        else:
            high_20 = np.nan
            low_20 = np.nan
        
        if np.isnan(high_20) or np.isnan(low_20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period 1d volume MA
        volume_ok = volume > 1.5 * vol_ma_20_1d_aligned[i]
        
        # ATR filter: current ATR > 0.5 * 1d ATR(14) (avoid low-volatility breakouts)
        # Approximate 6h ATR using recent price action (simplified: use price range)
        if i >= 2:
            atr_approx = np.max([
                abs(prices['high'].iloc[i] - prices['low'].iloc[i]),
                abs(prices['high'].iloc[i] - prices['close'].iloc[i-1]),
                abs(prices['low'].iloc[i] - prices['close'].iloc[i-1])
            ])
            atr_filter_ok = atr_approx > 0.5 * atr_14_1d_aligned[i]
        else:
            atr_filter_ok = False
        
        if position == 0:
            # Long breakout: price > Donchian high + volume + ATR + 12h uptrend
            if (price > high_20 and 
                volume_ok and 
                atr_filter_ok and 
                ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < Donchian low + volume + ATR + 12h downtrend
            elif (price < low_20 and 
                  volume_ok and 
                  atr_filter_ok and 
                  ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < 12h EMA34 (trend reversal) or Donchian mean reversion
            if price < ema_34_12h_aligned[i] or price < (high_20 + low_20) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > 12h EMA34 (trend reversal) or Donchian mean reversion
            if price > ema_34_12h_aligned[i] or price > (high_20 + low_20) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_1d_Donchian20_Breakout_VolumeATRFilter_v1"
timeframe = "6h"
leverage = 1.0