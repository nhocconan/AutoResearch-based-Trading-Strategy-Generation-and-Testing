#!/usr/bin/env python3
# 12h_Volume_Weighted_Camarilla_v1
# Hypothesis: Combines Camarilla pivot levels (from 1d) with volume-weighted price action to identify high-probability breakouts.
# Goes long when price closes above Camarilla H3 with above-average volume, short when below L3 with above-average volume.
# Uses 1w EMA50 as trend filter to avoid counter-trend trades. Designed for low trade frequency (<30/year) by requiring
# confluence of price level, volume confirmation, and trend alignment. Works in bull markets (follows trend) and bear markets
# (avoids counter-trend traps via weekly EMA filter).

name = "12h_Volume_Weighted_Camarilla_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Calculate Camarilla levels from previous 1d bar ---
    # Using formula: H4 = C + 1.1*(H-L)/2, H3 = C + 1.1*(H-L)/4, etc.
    # L4 = C - 1.1*(H-L)/2, L3 = C - 1.1*(H-L)/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    H3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    L3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    H4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    L4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # --- Volume confirmation (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # --- Trend filter: 50-period EMA on 1w close ---
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: price closes above H3 with volume, above weekly EMA50
            if (close[i] > H3_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price closes below L3 with volume, below weekly EMA50
            elif (close[i] < L3_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla level or loss of volume/momentum
            if position == 1:
                # Exit long: price closes below L3 or loses volume/momentum
                if (close[i] < L3_aligned[i] or 
                    vol_ratio[i] < 0.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above H3 or loses volume/momentum
                if (close[i] > H3_aligned[i] or 
                    vol_ratio[i] < 0.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals