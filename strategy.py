#!/usr/bin/env python3
"""
1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use 1d Camarilla R1/S1 breakout with 1w trend filter (EMA34) and volume confirmation. 
ATR-based trailing stop (2.5x ATR) to limit drawdowns. Target 30-100 trades over 4 years.
Works in bull/bear via trend filter and volatility-adjusted stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for 1w EMA34 trend filter
    
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # === 1w EMA34 Trend Filter ===
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 1d Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    
    # Calculate R1 and S1 for current bar using previous day's OHLC
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0  # for long trailing stop
    lowest_low_since_entry = 0.0    # for short trailing stop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_1w_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + above 1w EMA34 + volume
            if price > r1[i] and price > ema_1w_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below S1 + below 1w EMA34 + volume
            elif price < s1[i] and price < ema_1w_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            # ATR trailing stop: exit if price drops 2.5*ATR from highest high since entry
            if price < highest_high_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest low since entry
            if price > lowest_low_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "1d"
leverage = 1.0