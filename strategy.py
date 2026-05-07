#!/usr/bin/env python3

name = "4h_Camarilla_R1S1_Breakout_1dATR_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1_1d = pivot_1d + (range_1d * 1.0 / 6)
    S1_1d = pivot_1d - (range_1d * 1.0 / 6)
    R2_1d = pivot_1d + (range_1d * 2.0 / 6)
    S2_1d = pivot_1d - (range_1d * 2.0 / 6)
    
    # Calculate 1d ATR(14) for stop loss and volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 1.8x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.8 * vol_ma_20)
    
    # Trend filter: 4h close above/below 20-period EMA
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 4h to reduce trades
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above R1 with volume in uptrend
            if (close[i] > R1_1d_aligned[i] and 
                close[i] > ema_20[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: break below S1 with volume in downtrend
            elif (close[i] < S1_1d_aligned[i] and 
                  close[i] < ema_20[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: break below S1 or trend change or volatility spike
            if (close[i] < S1_1d_aligned[i] or 
                close[i] < ema_20[i] or 
                atr_1d_aligned[i] > 2.0 * np.nanmean(atr_1d_aligned[max(0, i-20):i])):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: break above R1 or trend change or volatility spike
            if (close[i] > R1_1d_aligned[i] or 
                close[i] > ema_20[i] or 
                atr_1d_aligned[i] > 2.0 * np.nanmean(atr_1d_aligned[max(0, i-20):i])):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 levels from daily timeframe act as strong support/resistance.
# Breaks above R1 with volume confirm bullish momentum; breaks below S1 confirm bearish.
# EMA20 filter ensures trades align with intermediate trend. Volatility spike exits protect
# against false breakouts. Works in both bull and bear markets by capturing breakouts
# in direction of daily trend. Target: 20-40 trades/year. Uses proven Camarilla structure
# with volume confirmation and volatility filter to reduce false signals. 4h timeframe
# balances signal quality and trade frequency. Avoids overtrading by requiring multiple
# confirmations and cooldown periods.