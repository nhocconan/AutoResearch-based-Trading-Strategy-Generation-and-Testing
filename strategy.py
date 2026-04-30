#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1d Supertrend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; Supertrend provides trend direction to avoid counter-trend trades.
# Long when Williams %R < -80 (oversold) and price > Supertrend (uptrend) with volume confirmation.
# Short when Williams %R > -20 (overbought) and price < Supertrend (downtrend) with volume confirmation.
# Exit when Williams %R crosses above -50 for longs or below -50 for shorts (mean reversion to midpoint).
# Williams %R is effective in ranging markets; Supertrend filter avoids whipsaws in strong trends.
# Volume confirmation reduces false signals. Timeframe: 6h as per experiment guidelines.

name = "6h_WilliamsR_MeanRev_1dSupertrend_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Supertrend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Supertrend parameters: ATR period=10, multiplier=3.0
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR using Wilder's smoothing (EMA with alpha=1/period)
    atr = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[1:atr_period+1])  # First ATR: average of first 'period' TR
        for i in range(atr_period, len(close_1d)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = np.full_like(close_1d, np.nan, dtype=float)
    direction = np.full_like(close_1d, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    if not np.isnan(atr[atr_period-1]):
        supertrend[atr_period-1] = upper_band[atr_period-1]
        direction[atr_period-1] = 1  # Start in uptrend
    
    for i in range(atr_period, len(close_1d)):
        if np.isnan(supertrend[i-1]) or np.isnan(direction[i-1]):
            continue
            
        # Supertrend logic
        if close_1d[i-1] > supertrend[i-1]:
            # Previous close was above previous Supertrend → uptrend
            direction[i] = 1
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            # Previous close was below previous Supertrend → downtrend
            direction[i] = -1
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Williams %R lookback, Supertrend initialization, volume MA
    start_idx = max(lookback, atr_period*2, 20)
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_supertrend = supertrend_aligned[i]
        curr_direction = direction_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND price > Supertrend (uptrend) AND volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > curr_supertrend and 
                curr_direction > 0 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < Supertrend (downtrend) AND volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < curr_supertrend and 
                  curr_direction < 0 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R crosses above -50 (mean reversion to midpoint)
            if curr_williams_r >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R crosses below -50 (mean reversion to midpoint)
            if curr_williams_r <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals