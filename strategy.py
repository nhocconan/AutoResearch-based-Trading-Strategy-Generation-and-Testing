#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 in uptrend (close > EMA50_12h) with volume > 1.5x average.
Short when price breaks below S1 in downtrend (close < EMA50_12h) with volume > 1.5x average.
Uses ATR-based stoploss and discrete sizing 0.25 to minimize fee churn.
Designed for BTC/ETH in both bull and bear markets by following 12h trend while using Camarilla levels for entries.
Target trades: 20-40/year (80-160 total over 4 years) to stay well below fee drag threshold.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1, R2, S2, R3, S3
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # R2 = close + (high-low)*1.1/6
    # R1 = close + (high-low)*1.1/12
    # S1 = close - (high-low)*1.1/12
    # S2 = close - (high-low)*1.1/6
    # S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    diff = (high_1d - low_1d) * 1.1
    R1 = close_1d + diff / 12
    S1 = close_1d - diff / 12
    R2 = close_1d + diff / 6
    S2 = close_1d - diff / 6
    R3 = close_1d + diff / 4
    S3 = close_1d - diff / 4
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 12h EMA (50), Camarilla (2), volume MA (20), ATR (14)
    start_idx = max(50, 2, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume spike
            long_signal = (high_val > R1_val) and (close_val > ema_50_12h_val) and vol_spike
            # Short: price breaks below S1 in downtrend with volume spike
            short_signal = (low_val < S1_val) and (close_val < ema_50_12h_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 or trend reversal or ATR stop
            if (close_val < S1_val or 
                close_val < ema_50_12h_val or
                close_val < entry_price - 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 or trend reversal or ATR stop
            if (close_val > R1_val or 
                close_val > ema_50_12h_val or
                close_val > entry_price + 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0