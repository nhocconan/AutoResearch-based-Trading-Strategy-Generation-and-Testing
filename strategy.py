#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_VolumeRegime_HTFTrend_v1
Hypothesis: 1h Camarilla R1/S1 breakouts filtered by 4h/1d trend alignment (EMA50/EMA200), volume regime (ATR-based), and session (08-20 UTC).
Uses discrete position sizing (0.0, ±0.20) to minimize fee churn. Targets 15-30 trades/year/symbol.
Works in bull/bear via multi-timeframe trend alignment and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === Previous day's Camarilla levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Align to 1h timeframe (use previous completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === HTF Trend: 4h EMA50 and 1d EMA200 ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    close_1d_ = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d_).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === ATR (14-period) for volume regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume regime: ATR > 20-period ATR mean (expanding volatility)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_regime = atr > 1.5 * atr_ma  # Expanding volatility regime
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup for 1d EMA200
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i])
            or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside session or not in expanding volatility regime
        if not (in_session[i] and vol_regime[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: price > Camarilla R1, 4h and 1d uptrend
            long_breakout = price > r1_aligned[i]
            long_trend_4h = price > ema_50_4h_aligned[i]
            long_trend_1d = price > ema_200_1d_aligned[i]
            
            # Short conditions: price < Camarilla S1, 4h and 1d downtrend
            short_breakout = price < s1_aligned[i]
            short_trend_4h = price < ema_50_4h_aligned[i]
            short_trend_1d = price < ema_200_1d_aligned[i]
            
            # Entry logic - require both timeframes to align
            if long_breakout and long_trend_4h and long_trend_1d:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_breakout and short_trend_4h and short_trend_1d:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: price closes below Camarilla S1 (support broken) OR trend fails
            if price < s1_aligned[i] or price < ema_50_4h_aligned[i] or price < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price closes above Camarilla R1 (resistance broken) OR trend fails
            if price > r1_aligned[i] or price > ema_50_4h_aligned[i] or price > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_VolumeRegime_HTFTrend_v1"
timeframe = "1h"
leverage = 1.0