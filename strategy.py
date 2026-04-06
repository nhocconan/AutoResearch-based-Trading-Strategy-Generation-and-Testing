#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 1.5x average
# Exit when Williams %R returns to -50 level
# Uses 6h timeframe to reduce trade frequency, targets 75-150 total trades over 4 years
# Williams %R identifies reversals, EMA50 filters for trend direction, volume confirms conviction
# Works in both bull/bear markets by trading pullbacks in the direction of higher timeframe trend

name = "6h_williamsr_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) - momentum oscillator
    # Values: -100 to 0, with -80 oversold, -20 overbought
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50 = ema_50.values
    
    # Align daily EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit when Williams %R returns to neutral (-50)
        if position == 1:  # long position
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Oversold + uptrend (price > EMA50) + volume
            if (williams_r[i] < -80 and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Overbought + downtrend (price < EMA50) + volume
            elif (williams_r[i] > -20 and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals