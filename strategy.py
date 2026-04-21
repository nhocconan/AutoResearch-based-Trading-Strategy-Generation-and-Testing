#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_V1
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h trend filter (price > 4h EMA34 for longs, < for shorts) and volume confirmation (>1.5x 20-period volume MA). Uses 1d ATR for dynamic stoploss. Target 15-37 trades/year (60-150 total over 4 years) on BTC/ETH. Uses 1h primary timeframe with 4h/1d HTF for regime and volatility filters.
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
    if len(df_4h) < 34 or len(df_1d) < 14:
        return np.zeros(n)
    
    # === 4h EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d ATR(14) for stoploss and volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # === 1h Indicators (primary timeframe) ===
    df_1h = prices  # Already 1h timeframe
    high = df_1h['high'].values
    low = df_1h['low'].values
    close = df_1h['close'].values
    volume = df_1h['volume'].values
    
    # Previous day's high/low for Camarilla calculation (using 1d data)
    # Camarilla R1 = Close + 1.1*(High-Low)/12
    # Camarilla S1 = Close - 1.1*(High-Low)/12
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    camarilla_range = (prev_high_1d - prev_low_1d)
    r1 = prev_close_1d + 1.1 * camarilla_range / 12
    s1 = prev_close_1d - 1.1 * camarilla_range / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) 
            or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + 4h uptrend
            if price > r1_aligned[i] and vol_ok and price > ema_34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + 4h downtrend
            elif price < s1_aligned[i] and vol_ok and price < ema_34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: stoploss or trend reversal
            stop_price = entry_price - 1.5 * atr_14_1d_aligned[i]
            if price < stop_price or price < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit conditions: stoploss or trend reversal
            stop_price = entry_price + 1.5 * atr_14_1d_aligned[i]
            if price > stop_price or price > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_V1"
timeframe = "1h"
leverage = 1.0