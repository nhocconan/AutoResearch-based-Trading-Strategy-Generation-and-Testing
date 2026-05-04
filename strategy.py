#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 resistance AND 1d bullish trend (close > EMA34) AND volume > 1.2x 20-period volume EMA
# Short when price breaks below Camarilla S1 support AND 1d bearish trend (close < EMA34) AND volume > 1.2x 20-period volume EMA
# Uses 1d EMA34 for stronger trend filter to reduce whipsaw, targeting 20-40 trades/year on 4h.
# Volume confirmation (1.2x) and tight Camarilla levels (R1/S1) reduce noise trades.
# Works in bull markets via longs in bullish 1d trend regime and bear markets via shorts in bearish 1d trend regime.

name = "4h_Camarilla_R1S1_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Align 1d trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Camarilla levels (R1, S1) from previous day's OHLC
    # Since we're on 4h timeframe, we'll use daily OHLC from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 calculation:
    # R1 = close + 1.1 * (high - low) / 2
    # S1 = close - 1.1 * (high - low) / 2
    camarilla_r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align prior day's Camarilla levels to 4h timeframe (wait for day to complete)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.2)  # Volume at least 1.2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND 1d bullish trend AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND 1d bearish trend AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S1 OR 1d trend turns bearish
            if (close[i] < camarilla_s1_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R1 OR 1d trend turns bullish
            if (close[i] > camarilla_r1_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals