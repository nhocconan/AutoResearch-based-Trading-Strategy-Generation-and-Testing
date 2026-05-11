#!/usr/bin/env python3
"""
4H_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Price breaking Camarilla R1/S1 levels with daily trend and volume confirmation
works in both bull and bear markets. Uses 4h timeframe with 1d trend filter to avoid
counter-trend trades. Targets 20-40 trades/year for low friction.
"""

name = "4H_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla levels from previous day ---
    # Calculate from previous day's OHLC (available at 4h bar close)
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Camarilla R1, S1 levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # --- 1d Trend filter (EMA34) ---
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Volume confirmation (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma.iloc[i])):
            if position != 0:
                # Hold position until exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions
        long_breakout = close[i] > R1_aligned[i]  # Price breaks above R1
        short_breakout = close[i] < S1_aligned[i]  # Price breaks below S1
        
        # Trend filter: only trade in direction of daily trend
        long_filter = close[i] > ema34_aligned[i]   # Above daily EMA34
        short_filter = close[i] < ema34_aligned[i]  # Below daily EMA34
        
        if position == 0:
            if long_breakout and long_filter and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and short_filter and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit if price breaks below S1 (reversal) or volume dries up
                if close[i] < S1_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if price breaks above R1 (reversal) or volume dries up
                if close[i] > R1_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals