#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA34 AND 1d volume > 1.5x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA34 AND 1d volume > 1.5x 20-period average.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short) or EMA trend reverses.
Uses 1d HTF for EMA34 trend and volume filter to reduce whipsaws in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams %R calculated as (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
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
    
    # Calculate 1d Williams %R (14-period) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_r = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high == lowest_low:
            williams_r[i] = -50  # Avoid division by zero
        else:
            williams_r[i] = ((highest_high - close_1d[i]) / (highest_high - lowest_low)) * -100
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume average for spike filter
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > EMA34 AND volume spike
            if wr < -80 and price > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < EMA34 AND volume spike
            elif wr > -20 and price < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 OR EMA trend turns down
                if wr > -50 or (i > start_idx and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 OR EMA trend turns up
                if wr < -50 or (i > start_idx and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_Reversal_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0