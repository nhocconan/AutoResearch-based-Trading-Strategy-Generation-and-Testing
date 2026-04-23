#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and ATR-based stoploss.
Long when price breaks above Donchian upper band AND 1d EMA34 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND 1d EMA34 is falling AND volume > 2.0x 20-period average.
Exit when price touches opposite Donchian band OR ATR trailing stop is hit (3*ATR from extreme).
Uses 1d HTF for EMA34 trend filter to avoid whipsaws. Volume spike filter reduces false breakouts.
Target: 75-200 total trades over 4 years (19-50/year) with discretionary sizing (0.25).
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
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 34, 20, 14)  # Donchian (20), EMA34 (34), volume MA (20), ATR (14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        up = upper[i]
        lo = lower[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA34 rising AND volume spike
            if price > up and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_high = price
            # Short: Break below Donchian lower AND EMA34 falling AND volume spike
            elif price < lo and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_low = price
        else:
            # Update extremes for trailing stop
            if position == 1:
                if price > highest_high:
                    highest_high = price
            elif position == -1:
                if price < lowest_low:
                    lowest_low = price
            
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower band OR EMA34 starts falling OR ATR stop hit
                if price < lo or ema_val < ema_34_aligned[i-1] or price <= highest_high - 3.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper band OR EMA34 starts rising OR ATR stop hit
                if price > up or ema_val > ema_34_aligned[i-1] or price >= lowest_low + 3.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0