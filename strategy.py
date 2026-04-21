#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1dTrend_VolumeSpike_ATRStop_v3
Hypothesis: 12h Camarilla pivot (R4/S4) breakouts filtered by 1d EMA200 trend and volume spike (>1.5x average).
Uses tighter timeframe (12h) to reduce trade frequency, EMA200 for strong trend filter, and volume confirmation to avoid false breakouts.
ATR trailing stop with 2.5x ATR distance. Designed for 12-30 trades/year per symbol.
Works in bull/bear via 1d trend alignment (EMA200) and volume confirmation to avoid whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for Camarilla, 1d for trend)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 5 or len(df_1d) < 200:
        return np.zeros(n)
    
    # === 12h OHLC for Camarilla calculation ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels (R4/S4) using previous completed 12h bar
    # R4 = Close + (High - Low) * 1.1 / 2
    # S4 = Close - (High - Low) * 1.1 / 2
    camarilla_range = (high_12h - low_12h) * 1.1 / 2
    r4 = close_12h + camarilla_range
    s4 = close_12h - camarilla_range
    
    # Align to 12h timeframe (use previous completed 12h bar)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # === 1d EMA200 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) 
            or np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr[i])):
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
            
            # Long conditions: price > 12h R4, 1d uptrend (price > EMA200), volume spike
            long_breakout = price > r4_aligned[i]
            long_trend = price > ema_200_1d_aligned[i]
            
            # Short conditions: price < 12h S4, 1d downtrend (price < EMA200), volume spike
            short_breakout = price < s4_aligned[i]
            short_trend = price < ema_200_1d_aligned[i]
            
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
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 12h S4 (support broken)
            elif price < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 12h R4 (resistance broken)
            elif price > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_1dTrend_VolumeSpike_ATRStop_v3"
timeframe = "12h"
leverage = 1.0