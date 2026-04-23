#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 1.5x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when Williams %R crosses -50 (mean reversion completion) or opposite extreme is hit.
Uses 1d HTF for EMA50 trend to avoid counter-trend trades in strong markets. Williams %R provides timely reversal signals in ranging/weak trending conditions.
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
    
    # Calculate Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 50, 20)  # Williams %R (14), EMA50 (50), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA50 AND volume spike
            if wr < -80 and price > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 1d EMA50 AND volume spike
            elif wr > -20 and price < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 OR reaches overbought (> -20)
                if wr > -50 or wr > -20:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 OR reaches oversold (< -80)
                if wr < -50 or wr < -80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extreme_Reversal_1dEMA50_Trend_VolumeConfirmation_WR50Exit"
timeframe = "6h"
leverage = 1.0