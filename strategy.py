#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R + 12h EMA Trend Filter with Volume Confirmation
# Hypothesis: Williams %R identifies overbought/oversold conditions. In strong trends (price > 12h EMA50),
# extreme readings signal continuation rather than reversal. Volume confirms institutional participation.
# Works in bull/bear by trading with the trend. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_williams_r_12h_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Williams %R (14-period)
    wr_period = 14
    highest_high = pd.Series(high).rolling(window=wr_period, min_periods=wr_period).max().values
    lowest_low = pd.Series(low).rolling(window=wr_period, min_periods=wr_period).min().values
    # Avoid division by zero
    diff = highest_high - lowest_low
    wr = np.where(diff != 0, -100 * (highest_high - close) / diff, -50.0)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma != 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R exits overbought OR trend weakens
            if wr[i] > -20 or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Williams %R exits oversold OR trend weakens
            if wr[i] < -80 or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation required
            if vol_ratio[i] > 1.5:
                # Strong uptrend: price above 12h EMA50 + Williams %R oversold (-80 to -100)
                if close[i] > ema_50_12h_aligned[i] and wr[i] <= -80:
                    position = 1
                    signals[i] = 0.25
                # Strong downtrend: price below 12h EMA50 + Williams %R overbought (-20 to 0)
                elif close[i] < ema_50_12h_aligned[i] and wr[i] >= -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals