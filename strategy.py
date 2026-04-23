#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above R4 and 12h EMA50 > price (bullish trend) with volume > 1.5x average.
Short when price breaks below S4 and 12h EMA50 < price (bearish trend) with volume > 1.5x average.
Exit on opposite Camarilla level break.
Camarilla levels from daily range provide robust support/resistance.
12h EMA50 filters for intermediate-term trend alignment.
Volume confirmation ensures breakout legitimacy.
Designed for 4h timeframe targeting 75-200 total trades over 4 years with controlled frequency.
Works in both bull and bear markets by only taking breakouts aligned with intermediate trend.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using prior bar's data)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Prior day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        # Typical price for pivot
        pivot = (phigh + plow + pclose) / 3
        range_val = phigh - plow
        
        # Camarilla R4 and S4 levels
        r4 = pivot + (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1 / 2)
        
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema_50_12h_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R4 AND 12h EMA50 > price (bullish) AND volume spike
            if (price > r4_val and ema_trend > price and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S4 AND 12h EMA50 < price (bearish) AND volume spike
            elif (price < s4_val and ema_trend < price and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S4
                if price < s4_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above R4
                if price > r4_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R4_S4_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0