#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
Long when price breaks above upper Donchian channel and 1d ATR(14) > 1.5x 50-period MA with volume > 1.3x average.
Short when price breaks below lower Donchian channel and 1d ATR(14) > 1.5x 50-period MA with volume > 1.3x average.
Exit on opposite Donchian break or ATR < 1.0x MA (low volatility regime).
Donchian channels provide clear trend structure, 1d ATR filter ensures breakouts occur in sufficient volatility environments.
Volume confirmation avoids false breakouts. Designed for 4h timeframe targeting 75-200 total trades over 4 years.
Works in both bull and bear markets by only taking breakouts when volatility is elevated (avoids choppy low-vol environments).
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
    
    # Load 1d data for ATR volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on 1d data
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d)
    
    # Calculate 50-period MA of ATR on 1d data
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR and ATR MA to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_val = atr_1d_aligned[i]
        atr_ma_val = atr_ma_1d_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 1d ATR > 1.5x ATR MA AND volume spike
            if (price > upper_val and atr_val > 1.5 * atr_ma_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND 1d ATR > 1.5x ATR MA AND volume spike
            elif (price < lower_val and atr_val > 1.5 * atr_ma_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR ATR < 1.0x ATR MA (low volatility)
                if (price < lower_val or atr_val < 1.0 * atr_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR ATR < 1.0x ATR MA (low volatility)
                if (price > upper_val or atr_val < 1.0 * atr_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_Volume_Breakout"
timeframe = "4h"
leverage = 1.0