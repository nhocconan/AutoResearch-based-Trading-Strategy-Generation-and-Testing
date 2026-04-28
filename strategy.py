#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot reversals with 1d trend filter and volume confirmation.
Uses Camarilla pivot levels (S1, S2, S3, R1, R2, R3) from daily high/low/close to identify
potential reversal zones. Entries occur when price rejects these levels in the direction
of the 1d EMA(34) trend, confirmed by volume spike (>2x 20-period average).
Designed for 12h timeframe with ~50-150 total trades over 4 years to minimize fee drag.
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
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Formula: R4 = close + ((high - low) * 1.1/2), R3 = close + ((high - low) * 1.1/4)
    #          R2 = close + ((high - low) * 1.1/6), R1 = close + ((high - low) * 1.1/12)
    #          S1 = close - ((high - low) * 1.1/12), S2 = close - ((high - low) * 1.1/6)
    #          S3 = close - ((high - low) * 1.1/4), S4 = close - ((high - low) * 1.1/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    rang = (high_1d - low_1d) * 1.1
    R1 = close_1d + rang / 12
    R2 = close_1d + rang / 6
    R3 = close_1d + rang / 4
    S1 = close_1d - rang / 12
    S2 = close_1d - rang / 6
    S3 = close_1d - rang / 4
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions: rejection of Camarilla levels in trend direction with volume
        # Long: price moves above S1/S2/S3 after being below (bounce off support)
        long_entry = (
            ((close[i] > S1_aligned[i] and close[i-1] <= S1_aligned[i]) or
             (close[i] > S2_aligned[i] and close[i-1] <= S2_aligned[i]) or
             (close[i] > S3_aligned[i] and close[i-1] <= S3_aligned[i])) and
            uptrend and
            volume_confirm[i]
        )
        
        # Short: price moves below R1/R2/R3 after being above (rejection at resistance)
        short_entry = (
            ((close[i] < R1_aligned[i] and close[i-1] >= R1_aligned[i]) or
             (close[i] < R2_aligned[i] and close[i-1] >= R2_aligned[i]) or
             (close[i] < R3_aligned[i] and close[i-1] >= R3_aligned[i])) and
            downtrend and
            volume_confirm[i]
        )
        
        # Exit conditions: opposite Camarilla level break or loss of trend
        long_exit = (
            (close[i] < S1_aligned[i]) or  # Broke below support
            (not uptrend)                  # Lost uptrend
        )
        
        short_exit = (
            (close[i] > R1_aligned[i]) or  # Broke above resistance
            (not downtrend)                # Lost downtrend
        )
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_CamarillaReversal_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0