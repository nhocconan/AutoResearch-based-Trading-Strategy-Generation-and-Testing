#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_VolumeSpike_12hTrend_ATRStop_v1
Hypothesis: 4h Camarilla R1/S1 breakouts filtered by 12h EMA34 trend and volume spike (>1.5x average).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn and target ~25 trades/year.
ATR-based trailing stop with 1.5x ATR distance. Works in bull/bear via 12h trend alignment and volume confirmation to avoid false breakouts.
Camarilla pivot levels provide statistically significant intraday support/resistance that price often respects.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for Camarilla, 12h for trend)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    if len(df_4h) < 10 or len(df_12h) < 34:
        return np.zeros(n)
    
    # === 4h Camarilla Pivot Points (R1, S1) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_4h + low_4h + close_4h) / 3.0
    # R1 = (2 * PP) - Low
    r1 = (2 * pp) - low_4h
    # S1 = (2 * PP) - High
    s1 = (2 * pp) - high_4h
    
    # Align to 4h timeframe (use previous completed 4h bar)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # === 12h EMA34 for HTF trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 1.5x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > 4h R1, 12h uptrend, volume spike
            long_breakout = price > r1_aligned[i]
            long_trend = price > ema_34_12h_aligned[i]
            
            # Short conditions: price < 4h S1, 12h downtrend, volume spike
            short_breakout = price < s1_aligned[i]
            short_trend = price < ema_34_12h_aligned[i]
            
            # Entry logic - ONLY enter on volume spike + trend alignment
            if long_breakout and long_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 4h S1 (support broken)
            elif price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 4h R1 (resistance broken)
            elif price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_12hTrend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0