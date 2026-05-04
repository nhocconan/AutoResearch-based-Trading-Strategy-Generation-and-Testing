#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator Jaw (teeth) AND 1d bullish trend (close > EMA34) AND volume > 1.5x 20-period volume EMA
# Short when price < Alligator Jaw (teeth) AND 1d bearish trend (close < EMA34) AND volume > 1.5x 20-period volume EMA
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) to identify trend absence/presence.
# In strong trends, Alligator lines are ordered and diverging; in ranging markets, they intertwine.
# We trade only when price is clearly above/below the Jaw, indicating a trending regime.
# Volume confirmation adds conviction. Designed for 12h timeframe to capture medium-term swings
# in both bull and bear markets, targeting 12-37 trades/year.

name = "12h_WilliamsAlligator_1dTrend_VolumeConfirm"
timeframe = "12h"
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
    
    # Align 1d trend to 12h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA (smoothed) of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    # SMMA (Smoothed Moving Average) is similar to EMA but with different smoothing
    # We'll use EMA as approximation for SMMA as it's commonly done
    median_price = (high + low) / 2
    
    # Jaw (13, 8)
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan
    
    # Teeth (8, 5)
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan
    
    # Lips (5, 3)
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan
    
    # For simplicity, we use the Jaw as the main trend indicator line
    # Price above Jaw = bullish bias, Price below Jaw = bearish bias
    alligator_jaw = jaw
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(alligator_jaw[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Alligator Jaw AND 1d bullish trend AND volume spike
            if (close[i] > alligator_jaw[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Alligator Jaw AND 1d bearish trend AND volume spike
            elif (close[i] < alligator_jaw[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Alligator Jaw OR 1d trend turns bearish
            if (close[i] < alligator_jaw[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Alligator Jaw OR 1d trend turns bullish
            if (close[i] > alligator_jaw[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals