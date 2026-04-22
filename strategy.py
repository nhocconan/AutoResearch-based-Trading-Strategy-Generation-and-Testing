#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with weekly trend filter, volume confirmation, and ATR-based exit.
Long when price breaks above 20-period Donchian upper band and weekly EMA50 rising with volume spike.
Short when price breaks below 20-period Donchian lower band and weekly EMA50 falling with volume spike.
Exit when price crosses opposite Donchian band or ATR-based trailing stop hit.
Designed for low trade frequency by requiring multiple confirmations and using higher timeframe trend.
Works in both bull and bear markets by following the weekly trend.
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
    
    # ATR for volatility filter and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # 50-period EMA on weekly close for trend
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above upper Donchian band with weekly EMA50 rising and volume spike
            if (close[i] > donchian_high[i] and 
                ema50_weekly_aligned[i] > ema50_weekly_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian band with weekly EMA50 falling and volume spike
            elif (close[i] < donchian_low[i] and 
                  ema50_weekly_aligned[i] < ema50_weekly_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Cross opposite band or ATR trailing stop
            exit_signal = False
            
            if position == 1:
                # Exit long: Cross below lower band or ATR stop
                if close[i] < donchian_low[i] or close[i] <= high_since_entry - 2.5 * atr[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Cross above upper band or ATR stop
                if close[i] > donchian_high[i] or close[i] >= low_since_entry + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
                
            # Track extreme prices for trailing stop
            if position == 1:
                if 'high_since_entry' not in locals():
                    high_since_entry = high[i]
                else:
                    high_since_entry = max(high_since_entry, high[i])
            else:
                if 'low_since_entry' not in locals():
                    low_since_entry = low[i]
                else:
                    low_since_entry = min(low_since_entry, low[i])
    
    return signals

name = "12H_DonchianBreakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0