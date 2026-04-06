#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R overbought/oversold with weekly trend filter and volume confirmation
# Long when weekly trend up (price > weekly EMA20), Williams %R oversold (< -80), and volume > 1.5x average
# Short when weekly trend down (price < weekly EMA20), Williams %R overbought (> -20), and volume > 1.5x average
# Exit when Williams %R returns to neutral range (-80 to -20) or volume drops
# Uses weekly trend to avoid counter-trend trades in strong moves, targets 50-150 total trades over 4 years
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend)

name = "6h_williamsr_weekly_trend_vol_v1"
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
    # Values: -100 to 0, oversold < -80, overbought > -20
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # Weekly trend filter: price vs weekly EMA20
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_values = weekly_ema.values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_up = close[i] > weekly_ema_aligned[i]
        weekly_down = close[i] < weekly_ema_aligned[i]
        
        # Exit conditions: Williams %R returns to neutral range OR volume drops
        if position == 1:  # long position
            if williams_r[i] > -80 or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] < -20 or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend alignment
            # Long: weekly trend up, Williams %R oversold, volume confirmation
            if weekly_up and williams_r[i] < -80 and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly trend down, Williams %R overbought, volume confirmation
            elif weekly_down and williams_r[i] > -20 and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
    
    return signals