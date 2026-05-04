#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 from below AND 1d bullish trend (close > EMA34) AND volume > 1.5x 20-period volume EMA
# Short when Williams %R(14) crosses below -20 from above AND 1d bearish trend (close < EMA34) AND volume > 1.5x 20-period volume EMA
# Uses Williams %R for mean reversion in extreme zones; 1d EMA34 for trend alignment; volume confirmation to reduce false signals.
# Designed for 6h timeframe: targets 12-37 trades/year (50-150 total over 4 years) with discrete position sizing (0.25) to minimize fee drag.
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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_1d = close_1d > ema_34_1d
    trend_bearish_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 6h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Williams %R(14) on 6h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R signals: cross above -80 (long) or below -20 (short)
    williams_long_signal = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_short_signal = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    # First bar cannot be a signal (no previous value)
    williams_long_signal[0] = False
    williams_short_signal[0] = False
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(williams_long_signal[i]) or np.isnan(williams_short_signal[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 AND 1d bullish trend AND volume spike
            if (williams_long_signal[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 AND 1d bearish trend AND volume spike
            elif (williams_short_signal[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 OR 1d trend turns bearish
            if (williams_r[i] < -50 and np.roll(williams_r, 1)[i] >= -50) or \
               trend_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 OR 1d trend turns bullish
            if (williams_r[i] > -50 and np.roll(williams_r, 1)[i] <= -50) or \
               trend_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals