#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1w_EMA50_Trend_VolumeSpike_v1
Hypothesis: On 12h timeframe, trade Camarilla R1/S1 breakouts only when aligned with 1w EMA50 trend and confirmed by volume spike (>2.0x 20-bar average). Uses ATR-based trailing stop. Camarilla levels from 1d provide institutional support/resistance, 1w EMA50 filters the major trend direction, volume confirms breakout strength. Designed for lower trade frequency (~12-37/year) to minimize fee drag while capturing sustained moves in both bull and bear markets.
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
    
    # Get 1d data for Camarilla levels (institutional support/resistance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter (major trend direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla R1 and S1 from previous 1d bar (HLC of daily)
    # Camarilla: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # ATR for stoploss and volatility filter
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: Break above R1, above 1w EMA50, with volume spike
            long_signal = (high_val > r1_level) and (close_val > ema_50_val) and vol_spike
            
            # Short: Break below S1, below 1w EMA50, with volume spike
            short_signal = (low_val < s1_level) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit: Close below EMA50 (trend change) OR trailing stop (2.5*ATR below high)
            if (close_val < ema_50_val) or (close_val < highest_since_entry - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit: Close above EMA50 (trend change) OR trailing stop (2.5*ATR above low)
            if (close_val > ema_50_val) or (close_val > lowest_since_entry + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1w_EMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0