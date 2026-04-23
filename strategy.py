#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume > 1.5x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume > 1.5x 20-period average.
Exit when Williams %R returns to -50 (mean reversion) or EMA34 trend reverses.
Uses 1d HTF for EMA34 to avoid 6h whipsaws. Williams %R identifies exhaustion in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if highest_high[i] == lowest_low[i]:
            williams_r[i] = -50  # Avoid division by zero
        else:
            williams_r[i] = ((highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])) * -100
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34)  # Williams %R (14), volume MA (20), EMA34 (34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA34 AND volume spike
            if wr < -80 and price > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 1d EMA34 AND volume spike
            elif wr > -20 and price < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R returns to -50 (mean reversion) OR EMA34 starts falling
                if wr >= -50 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R returns to -50 (mean reversion) OR EMA34 starts rising
                if wr <= -50 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0