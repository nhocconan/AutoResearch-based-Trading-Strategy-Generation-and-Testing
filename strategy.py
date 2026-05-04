#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power < previous Bear Power (strengthening bullish) AND 1d bullish trend (close > EMA50) AND volume > 1.5x 20-period volume EMA
# Short when Bear Power > 0 AND Bull Power < previous Bull Power (strengthening bearish) AND 1d bearish trend (close < EMA50) AND volume > 1.5x 20-period volume EMA
# Uses 1d EMA50 for trend filter to reduce whipsaw in bear markets, targeting 12-30 trades/year on 6h.
# Elder Ray measures bull/bear strength behind price moves; combined with trend filter captures sustainable moves.
# Works in bull markets via longs in bullish 1d trend regime and bear markets via shorts in bearish 1d trend regime.

name = "6h_ElderRay_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1d = close_1d > ema_50_1d
    trend_bearish_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 6h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13(close)
    # Bear Power = EMA13(close) - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Previous Bear Power and Bull Power for strength confirmation
    prev_bear_power = np.roll(bear_power, 1)
    prev_bull_power = np.roll(bull_power, 1)
    prev_bear_power[0] = np.nan
    prev_bull_power[0] = np.nan
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(prev_bull_power[i]) or np.isnan(prev_bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power weakening (decreasing) AND 1d bullish trend AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < prev_bear_power[i] and  # Bear Power decreasing = bulls gaining control
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 AND Bull Power weakening (decreasing) AND 1d bearish trend AND volume spike
            elif (bear_power[i] > 0 and 
                  bull_power[i] < prev_bull_power[i] and  # Bull Power decreasing = bears gaining control
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power > 0 (bears taking control) OR 1d trend turns bearish
            if (bear_power[i] > 0 or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 (bulls taking control) OR 1d trend turns bullish
            if (bull_power[i] > 0 or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals