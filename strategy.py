#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and ATR-based stoploss.
Long when price breaks above Donchian upper AND price > 1d EMA34 AND volume > 2.0x average.
Short when price breaks below Donchian lower AND price < 1d EMA34 AND volume > 2.0x average.
Exit via ATR trailing stop (3x ATR) or opposite Donchian breakout.
Donchian channels provide clear structure, 1d EMA34 ensures higher timeframe alignment,
volume spike confirms conviction, ATR stop manages risk. Designed for 4h timeframe
targeting 75-200 total trades over 4 years with low frequency to minimize fee drag.
Works in bull markets via breakouts and bear markets via short breakdowns.
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
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        atr_val = atr[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1d EMA34 AND volume spike
            if (price > upper_channel and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below lower Donchian AND price < 1d EMA34 AND volume spike
            elif (price < lower_channel and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                if price > highest_since_entry:
                    highest_since_entry = price
            else:  # position == -1
                if price < lowest_since_entry:
                    lowest_since_entry = price
            
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price drops below highest_since_entry - 3*ATR (trailing stop)
                # OR price breaks below lower Donchian (opposite signal)
                if (price < highest_since_entry - 3.0 * atr_val or price < lower_channel):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above lowest_since_entry + 3*ATR (trailing stop)
                # OR price breaks above upper Donchian (opposite signal)
                if (price > lowest_since_entry + 3.0 * atr_val or price > upper_channel):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0