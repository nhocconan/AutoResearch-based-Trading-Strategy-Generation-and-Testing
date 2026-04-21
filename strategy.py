#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA200 trend filter and volume spike (>2x 20-period MA). 
Long when price > R1 and above 4h EMA200 with volume spike. Short when price < S1 and below 4h EMA200 with volume spike.
ATR(14) stoploss (2.0x) and discrete sizing (0.20). Uses 4h HTF for trend alignment to capture major moves and reduce whipsaw.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag and improve test generalization.
Session filter: 08-20 UTC to avoid low-liquidity periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for EMA trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # === 4h OHLC for Camarilla pivot calculation (based on previous 4h bar) ===
    df_4h_open = df_4h['open'].values
    df_4h_high = df_4h['high'].values
    df_4h_low = df_4h['low'].values
    df_4h_close = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    range_4h = df_4h_high - df_4h_low
    r1_4h = df_4h_close + 0.275 * range_4h
    s1_4h = df_4h_close - 0.275 * range_4h
    
    # Align 4h Camarilla levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === 4h EMA200 for trend filter ===
    ema_200_4h = pd.Series(df_4h_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # === 1h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike filter (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) 
            or np.isnan(ema_200_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])
            or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_4h_aligned[i]
        s1 = s1_4h_aligned[i]
        ema_200 = ema_200_4h_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume spike: current volume > 2.0x average (avoid low-volume breakouts)
        volume_spike = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Enter only with volume spike and trend alignment
            long_condition = (price > r1) and (price > ema_200) and volume_spike
            short_condition = (price < s1) and (price < ema_200) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price below EMA)
            elif price < ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price above EMA)
            elif price > ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0