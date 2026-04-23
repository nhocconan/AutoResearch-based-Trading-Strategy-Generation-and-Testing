#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from 1h timeframe for precise entry, combined with
4h EMA50 trend filter to avoid counter-trend trades. Volume spike confirms breakout momentum.
Designed for 1h timeframe with tight entry conditions to target 15-37 trades/year.
Uses discrete position sizing (0.20) to minimize fee drag.
Works in bull markets via trend-following breakouts and in bear markets via short-side symmetry.
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
    
    # Calculate 4h EMA50 for trend filter (HTF direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot levels (R1, S1) based on previous 1h bar
    high_1h = df_4h['high'].values  # Use 4h high/low for 1h pivot calculation? No - need 1h data
    # Actually, we need to get 1h data for Camarilla calculation
    # But we only have 1h prices as primary timeframe - we can use rolling window on 1h data
    # Let's calculate 1h Camarilla using 1h OHLC directly
    
    # For 1h timeframe, we need to calculate Camarilla levels from previous 1h bar
    # We'll use rolling window on the 1h prices
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # We need previous bar's values, so shift by 1
    typical_price_prev = np.roll(typical_price, 1)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    typical_price_prev[0] = np.nan
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    
    range_prev = high_prev - low_prev
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_r1 = typical_price_prev + (range_prev * 1.1 / 12)
    camarilla_s1 = typical_price_prev - (range_prev * 1.1 / 12)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 1)  # need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 4h EMA50 direction
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend on 4h AND volume spike
            if close[i] > camarilla_r1[i] and trend_up and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND downtrend on 4h AND volume spike
            elif close[i] < camarilla_s1[i] and trend_down and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S1 for longs, R1 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S1
                if close[i] < camarilla_s1[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R1
                if close[i] > camarilla_r1[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0