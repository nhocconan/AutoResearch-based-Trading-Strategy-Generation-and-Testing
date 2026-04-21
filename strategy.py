#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_ATRStop
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike (>2.0x 20-period MA). 
Uses 4h for signal direction (trend + Camarilla) and 1h only for entry timing to reduce trade frequency. 
ATR(14) stoploss at 2.0x and minimum holding period of 3 bars to prevent churn. 
Designed for 15-37 trades/year (60-150 over 4 years) to minimize fee drag while capturing trending moves in both bull and bear markets via 4h trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend and Camarilla)
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
    
    # === 4h EMA34 for trend filter ===
    ema_34_4h = pd.Series(df_4h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) 
            or np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_4h_aligned[i]
        s1 = s1_4h_aligned[i]
        ema_34 = ema_34_4h_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume spike: current volume > 2.0x average (stricter for fewer trades)
        volume_spike = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Enter only with volume spike, trend alignment, and price closing beyond Camarilla level
            long_condition = (price > r1) and (price > ema_34) and volume_spike
            short_condition = (price < s1) and (price < ema_34) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.20 if position == 1 else -0.20
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below EMA)
                elif price < ema_34:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above EMA)
                elif price > ema_34:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_ATRStop"
timeframe = "1h"
leverage = 1.0