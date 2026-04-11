#!/usr/bin/env python3
# 12h_1w_camarilla_breakout_volume_v1
# Strategy: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from daily data act as strong support/resistance.
# Breakouts above R4 or below S4 with 1-week EMA trend confirmation and volume > 2x average
# capture institutional moves. Designed for low trade frequency (~15-30/year) to minimize fee drag.
# Works in bull markets via long breaks above R4 and bear markets via short breaks below S4.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Typical Camarilla: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    S1 = pivot - (range_hl * 1.1 / 12)
    S2 = pivot - (range_hl * 1.1 / 6)
    S3 = pivot - (range_hl * 1.1 / 4)
    S4 = pivot - (range_hl * 1.1 / 2)
    R1 = pivot + (range_hl * 1.1 / 12)
    R2 = pivot + (range_hl * 1.1 / 6)
    R3 = pivot + (range_hl * 1.1 / 4)
    R4 = pivot + (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    
    # 1-week EMA for trend filter (20-period)
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(S4_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_avg_20[i]
        
        # Breakout conditions
        breakout_up = close[i] > R4_aligned[i]
        breakdown_down = close[i] < S4_aligned[i]
        
        # Trend filter
        trend_bullish = close[i] > ema_20_1w_aligned[i]
        trend_bearish = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        # Long: Close above R4 AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Close below S4 AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite break of Camarilla levels
        elif position == 1 and close[i] < S4_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > R4_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals