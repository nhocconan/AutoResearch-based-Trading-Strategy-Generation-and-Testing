#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses discrete position sizing (0.25) to minimize fee churn. Donchian breakouts capture strong trends,
# while 1d EMA50 ensures alignment with higher-timeframe trend. Volume confirmation filters false breakouts.
# Target: 12-25 trades/year per symbol (50-100 total over 4 years) to avoid fee drag.
# Works in both bull and bear markets by only taking breakouts in the direction of the 1d trend.

name = "12h_Donchian20_1dEMA50_VolumeSpike_Trend"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels using lookback of 20 periods (20*12h = 10 days)
        start_idx = max(0, i - 20)
        highest_high = np.max(high[start_idx:i+1])
        lowest_low = np.min(low[start_idx:i+1])
        
        # Volume confirmation: current 12h volume > 1.5 x 20-period volume EMA
        vol_start = max(0, i - 20)
        vol_slice = volume[vol_start:i+1]
        if len(vol_slice) >= 20:
            vol_ema_20 = pd.Series(vol_slice).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
            volume_confirmed = volume[i] > (1.5 * vol_ema_20)
        else:
            volume_confirmed = False
        
        # 1d trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_1d_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band + volume confirmation + bullish 1d trend
            if (close[i] > highest_high and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + volume confirmation + bearish 1d trend
            elif (close[i] < lowest_low and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian lower band OR 1d trend turns bearish
            if close[i] < lowest_low or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Donchian upper band OR 1d trend turns bullish
            if close[i] > highest_high or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals