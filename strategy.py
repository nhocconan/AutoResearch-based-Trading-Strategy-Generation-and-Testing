#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R4/S4 breakouts on 6h with 1d EMA50 trend filter and volume confirmation.
R4/S4 represent the outer Camarilla levels - breakouts here with trend and volume have high continuation probability.
Uses 1d EMA50 for trend direction (stable daily trend) to filter breakouts.
Volume spike (>1.5x 20-bar average) confirms participation. Exits on reversion to Camarilla H3/L3 levels.
Discrete position sizing (0.25) minimizes fee churn. Target: 12-37 trades/year on 6h.
Works in bull markets (breakouts with trend) and bear markets (breakouts against trend via short signals).
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 1d data (based on previous bar's OHLC)
    # R4 = Close + (High-Low)*1.1/2
    # S4 = Close - (High-Low)*1.1/2
    # H3 = Close + (High-Low)*1.1/4
    # L3 = Close - (High-Low)*1.1/4
    camarilla_r4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    camarilla_h3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_l3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align HTF indicators to 6h timeframe (completed 1d bar lag)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d, additional_delay_bars=1)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d, additional_delay_bars=1)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50
    start_idx = 55
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 1d trend with volume confirmation
            # Long: price breaks above R4 in uptrend (close > EMA50)
            # Short: price breaks below S4 in downtrend (close < EMA50)
            long_signal = (close[i] > camarilla_r4_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s4_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla H3 (take profit at resistance)
            exit_signal = close[i] < camarilla_h3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla L3 (take profit at support)
            exit_signal = close[i] > camarilla_l3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0