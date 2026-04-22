#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot level touch with 1-day trend filter and volume confirmation.
Long when price touches or breaks above S3 with 1-day EMA34 rising and volume spike.
Short when price touches or breaks below R3 with 1-day EMA34 falling and volume spike.
Exit when price returns to mean (H4/L4) or trend reverses.
Camarilla levels provide institutional support/resistance; EMA34 filters trend direction; volume confirms participation.
Designed for low trade frequency by requiring confluence of price level, trend, and volume.
Works in both bull and bear markets by following the daily trend and using mean-reversion exits at pivot levels.
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
    
    # Calculate Camarilla levels for each bar using previous bar's OHLC
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # S3 = close - 1.125 * (high - low)
    # R3 = close + 1.125 * (high - low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    high_low_range = prev_high - prev_low
    
    H4 = prev_close + 1.5 * high_low_range
    L4 = prev_close - 1.5 * high_low_range
    H3 = prev_close + 1.25 * high_low_range
    L3 = prev_close - 1.25 * high_low_range
    S3 = prev_close - 1.125 * high_low_range
    R3 = prev_close + 1.125 * high_low_range
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(S3[i]) or np.isnan(R3[i]) or np.isnan(H4[i]) or np.isnan(L4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price at or below S3 with rising 1d EMA34 and volume spike
            if (close[i] <= S3[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price at or above R3 with falling 1d EMA34 and volume spike
            elif (close[i] >= R3[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to mean (H4/L4) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches H4 or 1d EMA34 turns down
                if close[i] >= H4[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches L4 or 1d EMA34 turns up
                if close[i] <= L4[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_S3R3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0