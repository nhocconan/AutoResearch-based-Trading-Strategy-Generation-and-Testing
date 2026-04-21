#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_Volume_ATRFilter_1h_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h with 4h trend filter (EMA34), 1d volume spike confirmation, and ATR volatility filter.
In bull markets: buy R1 breakouts above 4h EMA34. In bear markets: short S1 breakouts below 4h EMA34.
Volume spike and ATR filter ensure trades occur during high-momentum, volatile sessions (08-20 UTC).
Target: 20-40 trades/year per symbol (80-160 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 4h ATR(14) for volatility filter
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Load 1d data once for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # 1d volume 20-period EMA for spike detection
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Previous day's OHLC for Camarilla levels (1h timeframe entry)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Camarilla levels: R1, S1 (primary breakout levels)
    rang_1d = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + rang_1d * 1.0 / 12
    s1 = prev_close_1d - rang_1d * 1.0 / 12
    
    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(vol_ema_20_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume spike filter: current 1h volume > 2.0 * 1d volume EMA20 (scaled)
        # Approximate 1h volume vs daily: 1h volume > (daily EMA20 / 24) * 2.0
        vol_threshold = vol_ema_20_1d_aligned[i] / 24.0 * 2.0
        volume_ok = volume > vol_threshold
        
        # ATR filter: current ATR > 0.5 * 4h ATR(14) (scaled to 1h)
        # Approximate: 1h ATR > (4h ATR14 / (4*60/60)) * 0.5 = 4h ATR14 * 0.5 / 4
        atr_threshold = atr_14_4h_aligned[i] * 0.5 / 4.0
        # Use 1h ATR approximation from price range
        if i >= 1:
            tr_1h = max(
                prices['high'].iloc[i] - prices['low'].iloc[i],
                abs(prices['high'].iloc[i] - prices['close'].iloc[i-1]),
                abs(prices['low'].iloc[i] - prices['close'].iloc[i-1])
            )
        else:
            tr_1h = 0
        atr_approx = tr_1h  # simplified, using true range as ATR proxy
        atr_ok = atr_approx > atr_threshold
        
        if position == 0 and in_session:
            # Long: price breaks above R1 AND 4h uptrend AND volume spike AND ATR ok
            if (price > r1_aligned[i] and 
                ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and  # 4h EMA rising
                volume_ok and 
                atr_ok):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND 4h downtrend AND volume spike AND ATR ok
            elif (price < s1_aligned[i] and 
                  ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and  # 4h EMA falling
                  volume_ok and 
                  atr_ok):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price < 4h EMA34 (trend reversal) or time-based exit
            if price < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price > 4h EMA34 (trend reversal) or time-based exit
            if price > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_Volume_ATRFilter_1h_v1"
timeframe = "1h"
leverage = 1.0