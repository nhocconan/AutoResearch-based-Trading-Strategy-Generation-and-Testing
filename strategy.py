#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA50 (uptrend) AND volume > 1.5x 20-period MA.
Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA50 (downtrend) AND volume > 1.5x 20-period MA.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Designed for low trade frequency with mean reversion in ranging markets and trend filter to avoid counter-trend trades.
Williams %R identifies extreme price levels, daily trend ensures alignment with higher timeframe momentum.
Volume confirmation reduces false signals. Should work in both bull and bear markets by trading mean reversion
in the direction of the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # need EMA50, volume MA20, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA50 = uptrend, close < EMA50 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 4h volume > 1.5x 20-period MA (confirmation)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            if williams_r[i] < -80 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            elif williams_r[i] > -20 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50
                if williams_r[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0