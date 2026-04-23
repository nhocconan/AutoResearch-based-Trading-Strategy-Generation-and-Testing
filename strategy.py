#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and ATR-based stoploss.
Long when price breaks above Donchian upper band (20-period high) AND 1d EMA34 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band (20-period low) AND 1d EMA34 is falling AND volume > 2.0x 20-period average.
Exit via ATR trailing stop (3x ATR) or opposite Donchian breakout.
Uses 1d HTF for EMA34 trend to reduce whipsaws. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34, 20)  # Donchian (20), ATR (14), EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        atr_val = atr[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Donchian upper band AND EMA34 rising AND volume spike
            if price > donch_high and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_high = price
            # Short: Break below Donchian lower band AND EMA34 falling AND volume spike
            elif price < donch_low and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_low = price
        else:
            # Update highest/lowest for trailing stop
            if position == 1:
                if price > highest_high:
                    highest_high = price
            elif position == -1:
                if price < lowest_low:
                    lowest_low = price
            
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: ATR trailing stop OR price touches Donchian lower band
                if price <= highest_high - 3.0 * atr_val or price < donch_low:
                    exit_signal = True
            elif position == -1:
                # Short exit: ATR trailing stop OR price touches Donchian upper band
                if price >= lowest_low + 3.0 * atr_val or price > donch_high:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0