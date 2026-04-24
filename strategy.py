#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R(14) extreme reversal with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend direction (bull/bear filter).
- Williams %R(14): Identifies overbought (> -20) and oversold (< -80) conditions for mean reversion.
- Trend filter: Only take long signals when price > 12h EMA50 (bullish), short when price < 12h EMA50 (bearish).
- Volume confirmation: Current 6h volume > 2.0 * 20-period average volume to confirm momentum.
- Exit: Opposite Williams %R extreme (exit long when %R > -80, exit short when %R < -20).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets by buying oversold dips in uptrends, and in bear markets by selling overbought rallies in downtrends.
- Avoids choppy markets by requiring both trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Williams %R(14)
    if len(close) < 14:
        return np.zeros(n)
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume average for confirmation (20-period)
    if len(volume) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need 14 for Williams %R, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        
        # Trend filter: price > 12h EMA50 = bullish, price < 12h EMA50 = bearish
        bullish_trend = curr_close > ema_50_12h_aligned[i]
        bearish_trend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Exit conditions: opposite Williams %R extreme
        if position != 0:
            # Exit long: Williams %R > -80 (no longer oversold)
            if position == 1:
                if curr_williams_r > -80:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -20 (no longer overbought)
            elif position == -1:
                if curr_williams_r < -20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend and volume filters
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND bullish trend AND volume confirmation
            long_condition = (curr_williams_r < -80 and 
                            bullish_trend and
                            volume_confirm)
            
            # Short: Williams %R > -20 (overbought) AND bearish trend AND volume confirmation
            short_condition = (curr_williams_r > -20 and 
                             bearish_trend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_Extreme_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0