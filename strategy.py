#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMA) AND 1d bullish trend (close > EMA34) AND volume > 1.5x 20-period volume EMA
# Short when price < Alligator Jaw AND 1d bearish trend (close < EMA34) AND volume > 1.5x 20-period volume EMA
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs with specific shifts) to identify trend direction and avoid choppy markets
# Volume confirmation reduces false breakouts. 12h timeframe targets 12-37 trades/year to minimize fee drag.
# Works in bull markets via longs in bullish 1d trend regime and bear markets via shorts in bearish 1d trend regime.

name = "12h_WilliamsAlligator_1dTrend_VolumeConfirmation"
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
    
    # Williams Alligator indicators on 12h timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3)
    
    # Alligator conditions: price > Jaw for bullish, price < Jaw for bearish
    # Only trade when Alligator is aligned (Jaw > Teeth > Lips for bullish, Jaw < Teeth < Lips for bearish)
    bullish_alligator = (jaw > teeth) & (teeth > lips)
    bearish_alligator = (jaw < teeth) & (teeth < lips)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period for Alligator
        # Skip if any value is NaN
        if (np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND 1d bullish trend AND volume spike AND bullish Alligator alignment
            if (close[i] > jaw.iloc[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i] and 
                bullish_alligator.iloc[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw AND 1d bearish trend AND volume spike AND bearish Alligator alignment
            elif (close[i] < jaw.iloc[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i] and 
                  bearish_alligator.iloc[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Jaw OR 1d trend turns bearish OR Alligator loses bullish alignment
            if (close[i] < jaw.iloc[i] or 
                trend_bearish_aligned[i] > 0.5 or 
                not bullish_alligator.iloc[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Jaw OR 1d trend turns bullish OR Alligator loses bearish alignment
            if (close[i] > jaw.iloc[i] or 
                trend_bullish_aligned[i] > 0.5 or 
                not bearish_alligator.iloc[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals