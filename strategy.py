#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter (EMA20), 1d volatility filter (ATR20 > its 50-period average),
# and volume surge (1h volume > 1.5x its 20-period average). Enter on 20-period Donchian breakouts
# during active session (08:00-20:00 UTC). Use fixed position size 0.20. Designed for low trade
# frequency (target: 15-35 trades/year) by requiring multiple confluence factors. Works in bull/bear
# markets by trading breakouts with institutional volume confirmation and volatility expansion.

name = "1h_4h_1d_breakout_volatility_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient lookback for indicators
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08:00-20:00 UTC (inclusive 08:00, exclusive 20:00)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours < 20)
    
    # 4h EMA(20) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ATR(20) and its 50-period average for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 70:  # Need 20 for ATR + 50 for average
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_1d_avg = pd.Series(atr_20_1d).rolling(window=50, min_periods=50).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    atr_20_1d_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d_avg)
    
    # 1h Donchian breakout levels (20-period, excluding current bar)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1h volume average (20-period) for surge filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 60 to ensure all indicators are valid
    for i in range(60, n):
        # Skip if outside trading session (no new entries, but hold existing position)
        if not in_session[i]:
            signals[i] = 0.2 if position == 1 else (-0.2 if position == -1 else 0.0)
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_20_1d_aligned[i]) or 
            np.isnan(atr_20_1d_avg_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.2 if position == 1 else (-0.2 if position == -1 else 0.0)
            continue
        
        # Entry conditions
        trend_long = close[i] > ema_4h_aligned[i]
        trend_short = close[i] < ema_4h_aligned[i]
        vol_condition = atr_20_1d_aligned[i] > atr_20_1d_avg_aligned[i]
        vol_surge = volume[i] > 1.5 * vol_avg_20[i]
        breakout_long = high[i] > highest_high_20[i]
        breakout_short = low[i] < lowest_low_20[i]
        
        # State machine for position management
        if position == 0:
            if trend_long and vol_condition and vol_surge and breakout_long:
                position = 1
                signals[i] = 0.2
            elif trend_short and vol_condition and vol_surge and breakout_short:
                position = -1
                signals[i] = -0.2
        elif position == 1:
            # Exit long and enter short if short conditions met
            if trend_short and vol_condition and vol_surge and breakout_short:
                position = -1
                signals[i] = -0.2
            else:
                signals[i] = 0.2  # Hold long
        elif position == -1:
            # Exit short and enter long if long conditions met
            if trend_long and vol_condition and vol_surge and breakout_long:
                position = 1
                signals[i] = 0.2
            else:
                signals[i] = -0.2  # Hold short
    
    return signals