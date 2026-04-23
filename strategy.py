#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume spike + ATR stoploss for BTC/ETH/SOL.
Long when price breaks above Donchian(20) high AND volume > 2.0x 20-period volume MA.
Short when price breaks below Donchian(20) low AND volume > 2.0x 20-period volume MA.
Exit on ATR-based trailing stop (long: exit if price < highest_high_since_entry - 2.5*ATR;
short: exit if price > lowest_low_since_entry + 2.5*ATR).
Uses 1d HTF EMA34 trend filter to avoid counter-trend trades (only long when EMA34 rising,
only short when EMA34 falling).
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Volume spike threshold set high (2.0x) to minimize false breakouts and reduce trade frequency.
ATR trailing stop allows trends to run while controlling drawdown.
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        # Volume filter: 4h volume > 2.0x 20-period MA (high threshold to reduce false breakouts)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume filter AND EMA34 rising
            if close[i] > highest_high[i] and vol_filter and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            # Short: price breaks below Donchian low AND volume filter AND EMA34 falling
            elif close[i] < lowest_low[i] and vol_filter and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high[i])
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Long trailing stop: exit if price < highest_since_entry - 2.5*ATR
                if close[i] < highest_since_entry - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_bar = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                highest_since_entry = max(highest_since_entry, high[i])
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Short trailing stop: exit if price > lowest_since_entry + 2.5*ATR
                if close[i] > lowest_since_entry + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_bar = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_Breakout_VolumeSpike_ATRTrail_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0