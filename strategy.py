#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band squeeze breakout with 1-day volume confirmation
# Long when price breaks above upper BB (20,2) + 1-day volume > 1.5x 20-day average
# Short when price breaks below lower BB (20,2) + 1-day volume > 1.5x 20-day average
# Exit when price returns to middle BB (mean reversion)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Bollinger Bands for volatility-based entries and volume for confirmation
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_bb_squeeze_breakout_1d_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # 4-hour ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume average (20-day)
    vol_1d = df_1d['volume'].values
    vol_1d_s = pd.Series(vol_1d)
    vol_avg_20 = vol_1d_s.rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_avg_20  # 1.5x average volume
    
    # Align 1-day volume threshold to 4h
    vol_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_mid[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_threshold_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB (mean reversion)
            elif close[i] >= bb_mid[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB (mean reversion)
            elif close[i] <= bb_mid[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout with volume confirmation
            # Long: price breaks above upper BB + volume confirmation
            if close[i] > bb_upper[i] and volume[i] > vol_threshold_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower BB + volume confirmation
            elif close[i] < bb_lower[i] and volume[i] > vol_threshold_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals