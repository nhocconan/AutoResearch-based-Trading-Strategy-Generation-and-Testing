#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike + ATR stoploss.
Long when price breaks above Donchian upper AND price > 1d EMA34 AND volume > 2.0x average.
Short when price breaks below Donchian lower AND price < 1d EMA34 AND volume > 2.0x average.
Exit via ATR-based trailing stop: signal→0 when long position price < highest_high - 2.5*ATR or short position price > lowest_low + 2.5*ATR.
Donchian channels provide clear structure, 1d EMA34 filters for higher timeframe trend,
volume spike confirms conviction, ATR stop manages risk. Designed for 4h timeframe targeting 75-200 total trades over 4 years.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
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
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_up = highest_high
    donchian_low = lowest_low
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(donchian_up[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 1d EMA34 AND volume spike
            if (price > donchian_up[i] and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                highest_since_entry = price
            # Short: price breaks below Donchian lower AND price < 1d EMA34 AND volume spike
            elif (price < donchian_low[i] and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
                # Exit long: price drops below highest_since_entry - 2.5*ATR
                if price < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price)
                # Exit short: price rises above lowest_since_entry + 2.5*ATR
                if price > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0