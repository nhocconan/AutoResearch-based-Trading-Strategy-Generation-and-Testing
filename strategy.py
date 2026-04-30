#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Mean Reversion with 1d EMA50 trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels. Long when %R < -80 (oversold) AND price > 1d EMA50 (uptrend filter).
# Short when %R > -20 (overbought) AND price < 1d EMA50 (downtrend filter).
# Volume confirmation requires volume > 1.5x 20-bar average to ensure participation.
# Exit when Williams %R reverts to midpoint (-50) or opposite extreme.
# Williams %R is effective in ranging markets which dominate 2025 BTC/ETH action.
# 1d EMA50 filters for dominant long-term trend to avoid counter-trend entries.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_WilliamsR_MeanRev_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) AND uptrend (price > 1d EMA50) AND volume confirmation
            if (curr_wr < -80 and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend (price < 1d EMA50) AND volume confirmation
            elif (curr_wr > -20 and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R reverts to midpoint (-50) or reaches overbought (> -20)
            if curr_wr >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R reverts to midpoint (-50) or reaches oversold (< -80)
            if curr_wr <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals