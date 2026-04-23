#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
Long when price breaks above Donchian upper AND 12h EMA(50) rising AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower AND 12h EMA(50) falling AND volume > 1.5x 20-period average.
Exit when price returns to Donchian middle OR ATR trailing stop (2.0*ATR from extreme).
Donchian channels provide structural breakout levels; 12h EMA filters trend direction; volume confirms breakout strength.
Works in bull markets (breakouts up with rising EMA) and bear markets (breakouts down with falling EMA).
Target: ~25-35 trades/year on 4h timeframe with discrete sizing 0.30.
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
    
    # Calculate 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h Donchian(20)
    dc_period = 20
    upper_dc = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_dc = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    middle_dc = (upper_dc + lower_dc) / 2.0
    
    # 4h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 4h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(dc_period, 20)  # dc_period20, vol_ma20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = upper_dc[i]
        lower = lower_dc[i]
        middle = middle_dc[i]
        
        # EMA trend: rising if current > previous, falling if current < previous
        ema_rising = ema_val > ema_12h_aligned[i-1] if i > 0 else False
        ema_falling = ema_val < ema_12h_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: Price breaks above upper DC AND EMA rising AND volume spike
            if price > upper and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.30
                position = 1
                highest_since_entry = price
            # Short: Price breaks below lower DC AND EMA falling AND volume spike
            elif price < lower and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.30
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price returns to middle DC (mean reversion)
            if position == 1 and price < middle:
                exit_signal = True
            elif position == -1 and price > middle:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_MiddleExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0