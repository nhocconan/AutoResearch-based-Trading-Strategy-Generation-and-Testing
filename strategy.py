#!/usr/bin/env python3
name = "6h_AB_Swing_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # A-B Swing: A = prior swing low/high, B = current swing high/low
    # We'll use 5-period high/low for swing points
    roll_high_5 = pd.Series(high).rolling(window=5, min_periods=5).max().values
    roll_low_5 = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    # Swing detection: new high/low in last 3 bars
    swing_high = (roll_high_5 == high) & (high >= np.roll(high, 1)) & (high >= np.roll(high, 2)) & (high >= np.roll(high, 3))
    swing_low = (roll_low_5 == low) & (low <= np.roll(low, 1)) & (low <= np.roll(low, 2)) & (low <= np.roll(low, 3))
    
    # Most recent swing points (look back up to 20 bars)
    def find_most_recent(arr, start_idx):
        for i in range(start_idx, max(-1, start_idx - 20), -1):
            if arr[i]:
                return i
        return -1
    
    # Volume confirmation
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find most recent swing high/low
        swing_high_idx = find_most_recent(swing_high, i-1)
        swing_low_idx = find_most_recent(swing_low, i-1)
        
        if swing_high_idx == -1 or swing_low_idx == -1:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        swing_high_price = high[swing_high_idx]
        swing_low_price = low[swing_low_idx]
        
        # Avoid division by zero
        if swing_high_price == swing_low_price:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate where price is in the swing range (0 = low, 1 = high)
        price_in_range = (close[i] - swing_low_price) / (swing_high_price - swing_low_price)
        
        if position == 0:
            # Long: price near swing low (0-0.3) in daily uptrend with volume
            # Short: price near swing high (0.7-1.0) in daily downtrend with volume
            if price_in_range < 0.3 and ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1] and volume[i] > vol_ma_10[i] * 1.5:
                signals[i] = 0.25
                position = 1
            elif price_in_range > 0.7 and ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1] and volume[i] > vol_ma_10[i] * 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches swing high or trend changes
            if price_in_range > 0.7 or ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches swing low or trend changes
            if price_in_range < 0.3 or ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h A-B Swing trading with daily trend filter and volume confirmation
# - Identifies swing highs/lows using 5-period extremes
# - Enters long near swing lows (0-30% of range) in daily uptrends
# - Enters short near swing highs (70-100% of range) in daily downtrends
# - Requires volume confirmation (1.5x average) to avoid false signals
# - Exits when price reaches opposite swing or trend changes
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Position size 0.25 balances risk/reward while limiting trades
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Novel: Pure price action swing trading with institutional trend/volume filters
# - Avoids lagging indicators; uses actual swing points for precise entries
# - Daily EMA20 trend filter ensures alignment with higher timeframe momentum
# - Volume spike requirement reduces whipsaws in choppy markets