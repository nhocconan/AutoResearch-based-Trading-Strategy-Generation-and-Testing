#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume > 1.8x 20-period MA.
Short when Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume > 1.8x 20-period MA.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams %R provides mean-reversion signals, 1d EMA34 filters major trend, volume confirms reversal strength.
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
    
    # Calculate 4h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Williams %R (needs 14), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 4h volume > 1.8x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA34 AND volume filter
            if wr < -80 and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 1d EMA34 AND volume filter
            elif wr > -20 and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses above -50 (exit long) or below -50 (exit short)
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50
                if wr > -50:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0