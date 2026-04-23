#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation, and ATR-based stoploss.
Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x average.
Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x average.
Exit when price touches Donchian(10) opposite level OR ATR stoploss triggered (close-based).
Designed for 4h timeframe targeting 75-200 total trades over 4 years.
Works in bull markets via breakouts and in bear markets via short breakdowns.
Volume confirmation avoids low-conviction breakouts.
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
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels on 4h data
    donchian_len = 20
    donchian_exit_len = 10
    
    # Rolling max/min for Donchian(20)
    high_roll_max = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    low_roll_min = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Rolling max/min for Donchian(10) exit
    high_roll_max_exit = pd.Series(high).rolling(window=donchian_exit_len, min_periods=donchian_exit_len).max().values
    low_roll_min_exit = pd.Series(low).rolling(window=donchian_exit_len, min_periods=donchian_exit_len).min().values
    
    # Calculate ATR(14) for stoploss
    atr_len = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_len, min_periods=atr_len).mean().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(high_roll_max[i]) or 
            np.isnan(low_roll_min[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        donchian_high = high_roll_max[i]
        donchian_low = low_roll_min[i]
        donchian_high_exit = high_roll_max_exit[i]
        donchian_low_exit = low_roll_min_exit[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above Donchian(20) high AND price > 12h EMA50 AND volume spike
            if (price > donchian_high and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Donchian(20) low AND price < 12h EMA50 AND volume spike
            elif (price < donchian_low and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price touches Donchian(10) low OR ATR stoploss OR Donchian breakout reverse
                if (price <= donchian_low_exit or 
                    price <= entry_price - 2.5 * atr_val or
                    price < donchian_high):  # Reverse breakout
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price touches Donchian(10) high OR ATR stoploss OR Donchian breakout reverse
                if (price >= donchian_high_exit or 
                    price >= entry_price + 2.5 * atr_val or
                    price > donchian_low):  # Reverse breakout
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0