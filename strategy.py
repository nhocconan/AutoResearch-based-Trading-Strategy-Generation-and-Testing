#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMMA shifted 8) AND Alligator Mouth is open (Jaw > Teeth) AND 1w bullish trend AND volume > 1.5x 20-period volume EMA
# Short when price < Alligator Lips (8-period SMMA shifted 5) AND Alligator Mouth is open (Lips < Teeth) AND 1w bearish trend AND volume > 1.5x 20-period volume EMA
# Uses Williams Alligator to identify trending vs ranging markets, reducing whipsaw in choppy conditions.
# 1w EMA50 filter ensures we only trade with the major trend, improving performance in bear markets like 2025.
# Volume confirmation adds validity to breakouts. Targeting 7-25 trades/year on 1d.

name = "1d_WilliamsAlligator_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1w = close_1w > ema_50_1w
    trend_bearish_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate Williams Alligator components (using SMMA - smoothed moving average)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    # SMMA calculation (similar to Wilder's smoothing, equivalent to EMA with alpha=1/period)
    def smma(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate the shifted values that don't have enough history
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Alligator Mouth conditions
    # Mouth open bullish: Jaw > Teeth (trending up)
    # Mouth open bearish: Lips < Teeth (trending down)
    mouth_open_bullish = jaw > teeth
    mouth_open_bearish = lips < teeth
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND Mouth open bullish AND 1w bullish trend AND volume spike
            if (close[i] > jaw[i] and 
                mouth_open_bullish[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Lips AND Mouth open bearish AND 1w bearish trend AND volume spike
            elif (close[i] < lips[i] and 
                  mouth_open_bearish[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Teeth OR Mouth closes (Jaw <= Teeth) OR 1w trend turns bearish
            if (close[i] < teeth[i] or 
                not mouth_open_bullish[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Teeth OR Mouth closes (Lips >= Teeth) OR 1w trend turns bullish
            if (close[i] > teeth[i] or 
                not mouth_open_bearish[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals