#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray with 1-day mean reversion filter
# Long when Bull Power > 0, Bear Power < 0, and price < 1-day Bollinger Lower Band (20, 2)
# Short when Bear Power < 0, Bull Power < 0, and price > 1-day Bollinger Upper Band (20, 2)
# Exit when Elder Bull/Bear power crosses zero or price touches opposite Bollinger band
# Stoploss at 2 * ATR(14)
# Position size: 0.25
# Uses Elder Ray for momentum and Bollinger Bands for mean reversion entry timing
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_elder_ray_1d_bb_mr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Calculate Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR(14) for stoploss
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
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price touches upper Bollinger band or Bull Power <= 0
            elif close[i] >= bb_upper_aligned[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price touches lower Bollinger band or Bear Power >= 0
            elif close[i] <= bb_lower_aligned[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray with Bollinger Band mean reversion
            # Long: Bull Power > 0, Bear Power < 0, price < lower Bollinger band
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] < bb_lower_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power < 0, Bull Power < 0, price > upper Bollinger band
            elif bear_power[i] < 0 and bull_power[i] < 0 and close[i] > bb_upper_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals