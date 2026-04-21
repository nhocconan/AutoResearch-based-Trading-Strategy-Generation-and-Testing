#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot R1/S1 breakout with volume confirmation and trend filter.
Long when price breaks above R1 with volume > 1.5x average and close > daily EMA(34);
Short when price breaks below S1 with volume > 1.5x average and close < daily EMA(34).
Exit on opposite pivot touch or 2x ATR stop. Designed for 25-35 trades/year to minimize fee drag.
Works in bull markets via R1 breakouts and in bear via S1 breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous day's OHLC
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # Align pivots to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily EMA(34) for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ATR for stop (20-period on 4h)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current daily volume aligned to 4h
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_1d_current = vol_1d_aligned[i]
        
        if position == 0:
            # Enter long: break above R1 with volume surge and close > daily EMA34
            if (price_high > R1_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                price_close > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume surge and close < daily EMA34
            elif (price_low < S1_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                  price_close < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite pivot touch or 2x ATR stop
            exit_signal = False
            
            if position == 1:
                # Exit long: touch S1 OR price < entry - 2*ATR
                if price_low < S1_aligned[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use R1 as entry level for long
                    entry_level = R1_aligned[i-1] if i >= 1 else R1_aligned[0]
                    if price_close < entry_level - 2.0 * atr[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: touch R1 OR price > entry + 2*ATR
                if price_high > R1_aligned[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use S1 as entry level for short
                    entry_level = S1_aligned[i-1] if i >= 1 else S1_aligned[0]
                    if price_close > entry_level + 2.0 * atr[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume1.5x_EMA34Trend_ATR2x"
timeframe = "4h"
leverage = 1.0