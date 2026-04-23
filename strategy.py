#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 Breakout with 1d EMA34 Trend Filter and Volume Spike.
Long when price breaks above Camarilla R1 (1d) AND 1d EMA34 > 1d EMA34 previous candle (uptrend) AND 12h volume > 2.0x 20-period average volume.
Short when price breaks below Camarilla S1 (1d) AND 1d EMA34 < 1d EMA34 previous candle (downtrend) AND 12h volume > 2.0x 20-period average volume.
Exit when price returns to Camarilla pivot point (mean reversion) OR ATR trailing stop (2.0*ATR from extreme).
Camarilla levels provide intraday support/resistance; EMA34 filters for trending markets; volume confirms breakout strength.
Works in both bull (breakouts up) and bear (breakouts down) markets by capturing expansion phases at key levels.
Target: ~12-37 trades/year on 12h timeframe with discrete sizing 0.25.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    # Note: These are intraday levels calculated from same day's OHLC
    # For breakout signals, we use previous day's levels to avoid look-ahead
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    pivot_1d_prev = np.roll(pivot_1d, 1)
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    pivot_1d_prev[0] = np.nan
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_prev)
    
    # 12h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 12h trailing stop calculation
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
    start_idx = max(34, 20)  # EMA34, vol_ma20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_34_1d_curr = ema_34_1d_aligned[i]
        ema_34_1d_prev = ema_34_1d_aligned[i-1] if i > 0 else ema_34_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        pivot = pivot_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Uptrend: EMA34 rising
            uptrend = ema_34_1d_curr > ema_34_1d_prev
            # Downtrend: EMA34 falling
            downtrend = ema_34_1d_curr < ema_34_1d_prev
            
            # Long: Price breaks above R1 AND uptrend AND volume spike
            if price > r1 and uptrend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price breaks below S1 AND downtrend AND volume spike
            elif price < s1 and downtrend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
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
            
            # Primary exit: Price returns to pivot point (mean reversion)
            if position == 1 and price < pivot:
                exit_signal = True
            elif position == -1 and price > pivot:
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_PivotExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0