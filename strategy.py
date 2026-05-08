#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Alligator (Jaw/Teeth/Lips) to detect trend direction and strength.
# Adds 1d EMA(34) trend filter to avoid counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms momentum.
# Long when Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + volume spike.
# Short when Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + volume spike.
# Exit when Alligator lines re-cross or volume drops.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.
# Williams Alligator is effective in trending markets and avoids whipsaws in ranges.

name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator (13,8,5) smoothed with SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Blue line (13-period)
    teeth = smma(low, 8)   # Red line (8-period)
    lips = smma(close, 5)  # Green line (5-period)
    
    # Bullish alignment: Lips > Teeth > Jaw
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    # Bearish alignment: Lips < Teeth < Jaw
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_1d_up = ema_34_1d_aligned > np.roll(ema_34_1d_aligned, 1)
    trend_1d_down = ema_34_1d_aligned < np.roll(ema_34_1d_aligned, 1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bullish_alignment[i]) or np.isnan(bearish_alignment[i]) or
            np.isnan(trend_1d_up[i]) or np.isnan(trend_1d_down[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entry conditions
            if bullish_alignment[i] and trend_1d_up[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment[i] and trend_1d_down[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator re-cross or loss of trend/volume
            if not bullish_alignment[i] or not trend_1d_up[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator re-cross or loss of trend/volume
            if not bearish_alignment[i] or not trend_1d_down[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals