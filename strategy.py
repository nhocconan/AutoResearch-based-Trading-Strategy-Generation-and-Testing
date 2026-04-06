#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or volume drops
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in both bull/bear markets by fading extremes with trend filter

name = "12h_williamsr_1d_ema_vol_v10"
timeframe = "12h"
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
    # Values: -100 to 0, oversold below -80, overbought above -20
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    willr = willr.values
    
    # EMA50 from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema50 = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(willr[i]) or np.isnan(ema50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 level or volume drops
        if position == 1:  # long position
            if willr[i] > -50 or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if willr[i] < -50 or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Williams %R oversold (<-80) AND price above 1d EMA50 AND volume confirmation
            if (willr[i] < -80 and close[i] > ema50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (>-20) AND price below 1d EMA50 AND volume confirmation
            elif (willr[i] > -20 and close[i] < ema50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals