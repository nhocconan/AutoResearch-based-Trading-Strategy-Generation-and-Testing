#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2
Hypothesis: 4h Camarilla pivot (R1/S1) breakouts filtered by 1d EMA50 trend and 4h volume spike (>2x average).
Enter long when price breaks above 4h R1 with 1d uptrend and volume spike.
Enter short when price breaks below 4h S1 with 1d downtrend and volume spike.
Exit on ATR(14) trailing stop (2.5*ATR) or opposite level break.
Designed for low trade frequency (<30 trades/year) to minimize fee drag.
Works in bull/bear via 1d trend alignment and volume spike filter.
Version 2: Added 1d ATR regime filter to avoid whipsaw in ranging markets, reducing trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for pivots, 1d for trend and regime)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h Camarilla Pivot Levels (R1, S1) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_4h - low_4h) * 1.1 / 12.0
    r1_4h = close_4h + camarilla_range
    s1_4h = close_4h - camarilla_range
    
    # Align to 4h timeframe (use previous completed 4h bar)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === 1d EMA50 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d ATR(14) for regime filter: only trade when 1d ATR > 20-period average (avoid ranging) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d_arr, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d_arr, 1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_14_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_regime = atr_14_1d > atr_ma_20_1d  # True when volatility is elevated (trending market)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # === ATR (14-period) for stoploss (4h) ===
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
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) 
            or np.isnan(atr_regime_aligned[i])):
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
            
            # Regime filter: only trade when 1d ATR is elevated (trending market)
            reg_filter = atr_regime_aligned[i] > 0.5
            
            # Long conditions: price > 4h R1, 1d uptrend, volume spike, trending regime
            long_breakout = price > r1_4h_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < 4h S1, 1d downtrend, volume spike, trending regime
            short_breakout = price < s1_4h_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_spike and reg_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_spike and reg_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 4h S1 (support broken)
            elif price < s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 4h R1 (resistance broken)
            elif price > r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0