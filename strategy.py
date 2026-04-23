#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above Camarilla R1 AND 1d EMA34 rising AND volume > 2.0x 20-period MA.
Short when price breaks below Camarilla S1 AND 1d EMA34 falling AND volume > 2.0x 20-period MA.
Exit when price touches opposite Camarilla level (S1 for long, R1 for short) or 1d EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Camarilla levels provide precise intraday support/resistance, 1d EMA34 filters major trend,
volume confirms breakout strength. Designed to work in both bull (trend continuation) and bear
(mean reversion at extremes) markets via volume-spike filtered breakouts.
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
    
    # Calculate 4h Camarilla levels (using previous day's OHLC)
    # Camarilla levels are based on previous day's range
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    
    # Need to calculate based on daily OHLC, but we'll approximate using 4h data
    # For simplicity, we use 24-period (4h*6 = 1d) lookback for high/low/close
    lookback = 24  # 24 * 4h = 96h = 4 days, but we'll use 1 day approximation
    # Actually, Camarilla uses previous day's OHLC, so we need to shift by 1 day
    # We'll use 6-period lookback (6*4h = 24h = 1 day) for previous day's OHLC
    for i in range(6, n):
        # Previous day's OHLC: indices i-6 to i-1 (6 bars of 4h = 1 day)
        prev_high = np.max(high[i-6:i])
        prev_low = np.min(low[i-6:i])
        prev_close = close[i-1]  # Previous bar's close
        
        # Camarilla formulas
        range_val = prev_high - prev_low
        camarilla_R1[i] = prev_close + range_val * 1.1 / 12
        camarilla_S1[i] = prev_close - range_val * 1.1 / 12
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(6, 34, 20)  # Camarilla (needs 6), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        R1 = camarilla_R1[i]
        S1 = camarilla_S1[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 4h volume > 2.0x 20-period MA (strength filter)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA34 rising AND volume filter
            if price > R1 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND EMA34 falling AND volume filter
            elif price < S1 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S1 (opposite) OR EMA34 starts falling
                if price < S1 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R1 (opposite) OR EMA34 starts rising
                if price > R1 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0