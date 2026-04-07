#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray + 1-day regime + 1-week ATR filter
# Long when Bull Power > 0, Bear Power < 0, price > 200 EMA (1d), and weekly ATR rising
# Short when Bear Power < 0, Bull Power < 0, price < 200 EMA (1d), and weekly ATR rising
# Exit when Bull/Bear Power crosses zero
# Stoploss at 2.5 * ATR(14) (6h)
# Position size: 0.25
# Uses Elder Ray for momentum, 1-day EMA200 for trend filter, weekly ATR for volatility regime
# Target: 80-180 total trades over 4 years (20-45/year)

name = "6h_elder_ray_1d_ema200_1w_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1-week data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1-day EMA200
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1-week ATR(14) and its slope (rising/falling)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    # ATR slope: current - 3 periods ago
    atr_slope = np.diff(atr_1w, n=3, prepend=atr_1w[0])
    atr_slope_aligned = align_htf_to_ltf(prices, df_1w, atr_slope)
    
    # 6h Elder Ray components (13-period EMA as per standard)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR(14) for stoploss (6h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_slope_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power crosses below zero
            elif bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power crosses above zero
            elif bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals with EMA200 filter and rising ATR
            # Trend filter: price vs 200 EMA (1d)
            price_above_ema200 = close[i] > ema200_1d_aligned[i]
            price_below_ema200 = close[i] < ema200_1d_aligned[i]
            # Volatility filter: weekly ATR rising (positive slope)
            vol_filter = atr_slope_aligned[i] > 0
            
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > EMA200 AND ATR rising
            if bull_power[i] > 0 and bear_power[i] < 0 and price_above_ema200 and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power < 0 AND Bull Power < 0 AND price < EMA200 AND ATR rising
            elif bear_power[i] < 0 and bull_power[i] < 0 and price_below_ema200 and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals