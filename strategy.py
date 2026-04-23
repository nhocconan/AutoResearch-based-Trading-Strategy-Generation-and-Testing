#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above R1 AND price > 1d EMA34 AND volume > 2.0x average.
Short when price breaks below S1 AND price < 1d EMA34 AND volume > 2.0x average.
Exit on opposite Camarilla level touch (S1 for long, R1 for short) or ATR stoploss.
Uses discrete position sizing (0.30) to minimize fee churn. Targets 20-40 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance, effective in ranging markets,
while EMA34 filter ensures we only trade with the daily trend, reducing false breakouts.
Volume spike confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot calculation and EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for current 1d bar
    # Based on previous day's OHLC
    phigh_1d = np.roll(high_1d, 1)
    plow_1d = np.roll(low_1d, 1)
    pclose_1d = np.roll(close_1d, 1)
    # First bar: use same values
    phigh_1d[0] = high_1d[0]
    plow_1d[0] = low_1d[0]
    pclose_1d[0] = close_1d[0]
    
    pivot = (phigh_1d + plow_1d + pclose_1d) / 3.0
    range_1d = phigh_1d - plow_1d
    
    # Camarilla levels
    R1 = pivot + (range_1d * 1.0 / 12.0)
    S1 = pivot - (range_1d * 1.0 / 12.0)
    R2 = pivot + (range_1d * 2.0 / 12.0)
    S2 = pivot - (range_1d * 2.0 / 12.0)
    R3 = pivot + (range_1d * 3.0 / 12.0)
    S3 = pivot - (range_1d * 3.0 / 12.0)
    R4 = pivot + (range_1d * 4.0 / 12.0)
    S4 = pivot - (range_1d * 4.0 / 12.0)
    
    # Calculate EMA34 on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 1d timeframe
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma_1d_4h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(14) on 1d for stoploss
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema34_1d_4h[i]) or np.isnan(vol_ma_1d_4h[i]) or
            np.isnan(atr_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_1d_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 1d EMA34 AND volume spike
            if (price > R1_4h[i] and 
                price > ema34_1d_4h[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: price breaks below S1 AND price < 1d EMA34 AND volume spike
            elif (price < S1_4h[i] and 
                  price < ema34_1d_4h[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches S1 (opposite level) OR ATR stoploss
                if price <= S1_4h[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_1d_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches R1 (opposite level) OR ATR stoploss
                if price >= R1_4h[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_1d_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0