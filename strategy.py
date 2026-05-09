#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day ATR-based volatility breakout with 1-day trend filter and volume confirmation.
# Enters long when price breaks above previous day's high + ATR(14) with daily uptrend and volume spike.
# Enters short when price breaks below previous day's low - ATR(14) with daily downtrend and volume spike.
# Exits on trend reversal or price crossing the opposite day's extreme.
# Designed to work in both bull and bear markets by aligning with daily trend. Target: 20-50 trades/year to minimize fee drag.

name = "4h_ATRBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR, high/low, and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) - smoothed moving average of TR
    atr_14 = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 15:  # Need at least 15 values for first ATR
        atr_14[14] = np.nanmean(tr[1:15])  # First ATR is average of first 14 TR values
        for i in range(15, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate EMA20 on 1d close for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Previous day's high and low for breakout levels
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Breakout levels: prev day's high/low ± ATR(14)
    breakout_high = prev_high_1d + atr_14
    breakout_low = prev_low_1d - atr_14
    
    # Align all 1d indicators to 4h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    breakout_high_aligned = align_htf_to_ltf(prices, df_1d, breakout_high)
    breakout_low_aligned = align_htf_to_ltf(prices, df_1d, breakout_low)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for EMA20 (1d) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(breakout_high_aligned[i]) or 
            np.isnan(breakout_low_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1d_val = ema20_1d_aligned[i]
        breakout_high_val = breakout_high_aligned[i]
        breakout_low_val = breakout_low_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above breakout_high + 1d uptrend + volume spike
            if close[i] > breakout_high_val and close[i] > ema20_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below breakout_low + 1d downtrend + volume spike
            elif close[i] < breakout_low_val and close[i] < ema20_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below breakout_low or 1d trend turns down
            if close[i] < breakout_low_val or close[i] < ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above breakout_high or 1d trend turns up
            if close[i] > breakout_high_val or close[i] > ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals