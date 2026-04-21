#!/usr/bin/env python3
"""
1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeATRFilter_V2
Hypothesis: Daily Camarilla R1/S1 breakouts with weekly trend filter (price > weekly EMA34 for longs, < for shorts) and volume confirmation (>1.5x 20-period volume MA). 
Camarilla levels identify key intraday support/resistance; weekly EMA34 filters for higher-timeframe trend alignment. 
Volume confirmation reduces false breakouts. ATR-based stoploss manages risk. 
Target 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL.
Uses 1d primary timeframe with 1w HTF for Camarilla calculation and EMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for Camarilla and EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w Camarilla Pivot Points (R1, S1) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1w + low_1w + close_1w) / 3
    # Camarilla width
    camarilla_width = (high_1w - low_1w) * 1.1 / 12
    # R1 and S1 levels
    r1 = close_1w + camarilla_width * 1.1
    s1 = close_1w - camarilla_width * 1.1
    
    # Align HTF levels to LTF (no extra delay needed for pivot points)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === 1w EMA34 for trend filter ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d Indicators (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + weekly uptrend
            if price > r1_aligned[i] and vol_ok and price > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + weekly downtrend
            elif price < s1_aligned[i] and vol_ok and price < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: stoploss, trend reversal, or volatility collapse
            stoploss = entry_price - 2.5 * atr[i]
            if price < stoploss or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: stoploss, trend reversal, or volatility collapse
            stoploss = entry_price + 2.5 * atr[i]
            if price > stoploss or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeATRFilter_V2"
timeframe = "1d"
leverage = 1.0