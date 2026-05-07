#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour weekly Camarilla pivot breakout with daily volume confirmation and weekly trend filter.
# Long when: Close > Camarilla R4 (from weekly) AND weekly EMA34 rising AND daily volume > 2.0 * EMA20(volume).
# Short when: Close < Camarilla S4 (from weekly) AND weekly EMA34 falling AND daily volume > 2.0 * EMA20(volume).
# Exit when price crosses back below/above the weekly EMA8.
# Uses weekly Camarilla levels for strong structure, weekly EMA34 for trend, and daily volume for confirmation.
# Designed for very low trade frequency (<15/year) to minimize fee drag and improve generalization in bear markets.
name = "12h_WeeklyCamarilla_R4S4_VolumeTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Camarilla levels and EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # Using prior week's values (shifted by 1 to avoid look-ahead)
    camarilla_r4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_s4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1w, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1w, dtype=bool)
    ema_34_rising[1:] = ema_34_1w[1:] > ema_34_1w[:-1]
    ema_34_falling[1:] = ema_34_1w[1:] < ema_34_1w[:-1]
    
    # Align weekly data to 12h timeframe (waits for weekly bar to close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_falling)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ema_20_1d)
    
    # Align daily volume spike to 12h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Weekly EMA8 for exit
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for weekly indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema_8_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Weekly Camarilla R4 AND weekly EMA34 rising AND daily volume spike
            long_condition = (close[i] > camarilla_r4_aligned[i]) and ema_34_rising_aligned[i] and volume_spike_1d_aligned[i]
            # Short: Close < Weekly Camarilla S4 AND weekly EMA34 falling AND daily volume spike
            short_condition = (close[i] < camarilla_s4_aligned[i]) and ema_34_falling_aligned[i] and volume_spike_1d_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < Weekly EMA8
            if close[i] < ema_8_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > Weekly EMA8
            if close[i] > ema_8_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals