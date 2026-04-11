#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_volume_trend_v1
# Strategy: 12h Camarilla pivot breakout with volume confirmation and 1d EMA50 trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Uses Camarilla pivot levels from 1d timeframe for breakout entries,
# requiring volume confirmation (>1.5x average volume) and filtered by 1d EMA50 trend alignment.
# Works in both bull and bear markets by following the higher timeframe trend (1d).
# Targets 50-150 total trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla and EMA50
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from 1d data
    # Camarilla levels: Close ± (High - Low) * multipliers
    range_1d = high_1d - low_1d
    camarilla_H4 = close_1d + range_1d * 1.1/2  # Resistance 4
    camarilla_L4 = close_1d - range_1d * 1.1/2  # Support 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_H4_aligned[i]) or 
            np.isnan(camarilla_L4_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = price_close > camarilla_H4_aligned[i-1]  # Break above previous period H4
        breakout_down = price_close < camarilla_L4_aligned[i-1]  # Break below previous period L4
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Camarilla breakout up with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1d
        
        # Short: Camarilla breakout down with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1d
        
        # Exit when price returns to the middle of the Camarilla range
        camarilla_mid = (camarilla_H4_aligned[i] + camarilla_L4_aligned[i]) / 2
        exit_long = position == 1 and price_close < camarilla_mid
        exit_short = position == -1 and price_close > camarilla_mid
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals