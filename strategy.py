#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 from oversold AND 1d bullish trend (close > EMA34) AND volume > 1.3x 20-period volume EMA
# Short when Williams %R(14) crosses below -20 from overbought AND 1d bearish trend (close < EMA34) AND volume > 1.3x 20-period volume EMA
# Williams %R identifies exhaustion points; 1d EMA34 ensures trading with higher timeframe trend; volume confirmation reduces false signals.
# Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25) to minimize fee drag.
# Works in bull markets via longs in bullish 1d trend and bear markets via shorts in bearish 1d trend.

name = "6h_WilliamsR_Extreme_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_1d = close_1d > ema_34_1d
    trend_bearish_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 6h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate Williams %R crossovers
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    
    # Long signal: Williams %R crosses above -80 (from below)
    williams_r_long_entry = (williams_r > -80) & (williams_r_prev <= -80)
    # Short signal: Williams %R crosses below -20 (from above)
    williams_r_short_entry = (williams_r < -20) & (williams_r_prev >= -20)
    
    # Align Williams %R signals to 6h timeframe (already on 6h, but ensure proper alignment)
    williams_r_long_aligned = align_htf_to_ltf(prices, prices, williams_r_long_entry.astype(float))
    williams_r_short_aligned = align_htf_to_ltf(prices, prices, williams_r_short_entry.astype(float))
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.3)  # Volume at least 1.3x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(williams_r_long_aligned[i]) or np.isnan(williams_r_short_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 AND 1d bullish trend AND volume spike
            if (williams_r_long_aligned[i] > 0.5 and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 AND 1d bearish trend AND volume spike
            elif (williams_r_short_aligned[i] > 0.5 and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR 1d trend turns bearish
            if (williams_r[i] > -20 or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR 1d trend turns bullish
            if (williams_r[i] < -80 or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals