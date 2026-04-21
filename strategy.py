#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1w_Trend_VolumeSpike
Hypothesis: On 1d timeframe, enter long when price breaks above Camarilla R1 with 1w EMA34 uptrend and volume > 2x 20-period MA; enter short when price breaks below S1 with 1w EMA34 downtrend and volume confirmation. Uses discrete sizing (0.30) and ATR(14) stoploss (2.0x). Designed for low trade frequency (<25/year) to minimize fee drag while capturing major trend moves in both bull and bear markets via 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    
    # Align 1d Camarilla levels to 1d timeframe (no alignment needed, but using helper for consistency)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1w EMA34 for trend filter ===
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume regime filter (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema_34 = ema_34_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume regime: current volume > 2.0x average (strict filter to reduce trades)
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Enter only with volume confirmation and trend alignment
            long_condition = (price > r1) and (price > ema_34) and volume_confirmed
            short_condition = (price < s1) and (price < ema_34) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price below EMA)
            elif price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price above EMA)
            elif price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1w_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0