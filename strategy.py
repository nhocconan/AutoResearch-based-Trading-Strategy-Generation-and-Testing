#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R4 (1d Camarilla) AND price > 1d EMA34 (uptrend) AND volume > 2.0x average.
Short when price breaks below S4 (1d Camarilla) AND price < 1d EMA34 (downtrend) AND volume > 2.0x average.
Exit when price reverts to Camarilla pivot point (PP) or trend reverses (price crosses 1d EMA34).
Uses tighter R4/S4 levels to reduce false breakouts and overtrading. Target: 20-40 trades/year.
Includes ATR-based stoploss for risk management. Works in bull/bear via trend filter + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot levels and EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: PP = (H+L+C)/3, R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Calculate EMA34 for 1d trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr[0] = np.nan  # First value undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R4 AND price > 1d EMA34 (uptrend) AND volume > 2.0x average
            if (price > r4_val and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S4 AND price < 1d EMA34 (downtrend) AND volume > 2.0x average
            elif (price < s4_val and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot point OR price breaks below 1d EMA34 (trend reversal) OR stoploss hit
                if price <= pp_val or price < ema34_val or price < entry_price - 2.0 * atr_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot point OR price breaks above 1d EMA34 (trend reversal) OR stoploss hit
                if price >= pp_val or price > ema34_val or price > entry_price + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R4_S4_Breakout_1dEMA34_Volume_ATR"
timeframe = "4h"
leverage = 1.0