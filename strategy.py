#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_ATRStop_v1
Hypothesis: 1h Camarilla pivot (R1/S1) breakouts filtered by 4h EMA34 trend and volume spike (>2x average).
Uses 4h trend for signal direction, 1h only for entry timing. Discrete sizing (0.0, ±0.20) to minimize fee churn.
ATR trailing stop with 2.0x ATR distance. Session filter (08-20 UTC) reduces noise.
Designed for 15-35 trades/year per symbol in both bull and bear markets via trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend, pivots also from 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h EMA34 for HTF trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 4h Camarilla Pivot Levels (R1, S1) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    camarilla_range = (high_4h - low_4h) * 1.1 / 12.0
    r1_4h = close_4h + camarilla_range
    s1_4h = close_4h - camarilla_range
    
    # Align Camarilla levels to 1h timeframe (use previous completed 4h bar)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) 
            or np.isnan(s1_4h_aligned[i]) or np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 2x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 2.0 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > 4h R1, 4h uptrend, volume spike
            long_breakout = price > r1_4h_aligned[i]
            long_trend = price > ema_34_4h_aligned[i]
            
            # Short conditions: price < 4h S1, 4h downtrend, volume spike
            short_breakout = price < s1_4h_aligned[i]
            short_trend = price < ema_34_4h_aligned[i]
            
            # Entry logic - ONLY enter on volume spike + trend alignment
            if long_breakout and long_trend and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 4h S1 (support broken)
            elif price < s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 4h R1 (resistance broken)
            elif price > r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_ATRStop_v1"
timeframe = "1h"
leverage = 1.0