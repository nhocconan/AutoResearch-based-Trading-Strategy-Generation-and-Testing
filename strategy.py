#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeSpike_HTFTrend_ATRStop
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation (>2.0x 20-bar volume MA) and 1d EMA50 trend filter.
ATR-based trailing stop (3x ATR) to manage risk. Camarilla pivots from 1d provide strong intraday levels.
Volume spike confirms breakout strength. EMA50 trend filter ensures alignment with higher timeframe.
Position size 0.25 balances risk/return. Target 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # We need to get the previous day's OHLC for each 4h bar
    # Since we're on 4h timeframe, we can use the 1d data to get daily OHLC
    # For each 4h bar, we use the previous completed 1d bar's OHLC
    
    # Get daily OHLC arrays
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    daily_range = daily_high - daily_low
    camarilla_R1 = daily_close + daily_range * 1.1 / 12
    camarilla_S1 = daily_close - daily_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) 
            or np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume confirmation + price > 1d EMA50
            if price > camarilla_R1_aligned[i] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below Camarilla S1 + volume confirmation + price < 1d EMA50
            elif price < camarilla_S1_aligned[i] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 3*ATR from highest since entry
            if price < highest_since_entry - 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 3*ATR from lowest since entry
            if price > lowest_since_entry + 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_HTFTrend_ATRStop"
timeframe = "4h"
leverage = 1.0