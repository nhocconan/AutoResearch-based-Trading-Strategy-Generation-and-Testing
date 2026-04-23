#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
- Long when price breaks above Camarilla R3 AND close > 1d EMA50 AND volume > 2.0x 20-period average
- Short when price breaks below Camarilla S3 AND close < 1d EMA50 AND volume > 2.0x 20-period average
- Exit when price crosses Camarilla H4/L4 levels (mean reversion)
- Uses 1d EMA50 for HTF trend alignment to avoid counter-trend entries
- Camarilla levels calculated from prior 1d OHLC for structure-based entries
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in both bull/bear markets: trend filter prevents counter-trend entries, structure provides clear levels
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
    
    # Get 1d data for EMA50 trend filter and Camarilla calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # Camarilla: based on prior day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # R3 = Close + 1.1*(High-Low)/2, S3 = Close - 1.1*(High-Low)/2
    # H4 = Close + 1.1*(High-Low), L4 = Close - 1.1*(High-Low)
    range_1d = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * range_1d
    camarilla_l4 = prev_close - 1.1 * range_1d
    camarilla_h3 = prev_close + 1.1 * range_1d / 2
    camarilla_l3 = prev_close - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Need 20 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Break above prior period's H3 (R3)
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Break below prior period's L3 (S3)
        
        # Trend filter (using 1d EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation (stricter threshold)
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Camarilla H4/L4 levels (mean reversion to mean)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below H4 (or reaches L4 for deeper mean reversion)
                if close[i] < camarilla_h4_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above L4 (or reaches H4 for deeper mean reversion)
                if close[i] > camarilla_l4_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0