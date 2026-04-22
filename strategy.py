#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with weekly trend filter and volume confirmation.
# The Alligator (Jaw/Teeth/Lips) acts as a trend indicator: when Lips > Teeth > Jaw = bullish,
# Lips < Teeth < Jaw = bearish. Trades are taken in direction of weekly trend with volume spike.
# Designed for daily timeframe to capture multi-week swings with very low frequency.
# Target: 10-25 trades/year per symbol (40-100 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter via Alligator
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Williams Alligator on weekly: Jaw=13-period SMMA(8), Teeth=8-period SMMA(5), Lips=5-period SMMA(3)
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1w, 13)  # 13-period, shifted 8 bars forward
    teeth = smma(close_1w, 8)  # 8-period, shifted 5 bars forward
    lips = smma(close_1w, 5)   # 5-period, shifted 3 bars forward
    
    # Shift to avoid look-ahead (Alligator lines are plotted forward)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Trend: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
    bullish = (lips > teeth) & (teeth > jaw)
    bearish = (lips < teeth) & (teeth < jaw)
    
    # Volume spike filter (20-period on daily)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Align indicators to daily timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish + volume spike
            if bullish_aligned[i] > 0.5 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + volume spike
            elif bearish_aligned[i] > 0.5 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator changes direction or volume drops
            if position == 1:
                if bullish_aligned[i] <= 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bearish_aligned[i] <= 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_WeeklyTrend_Volume_Spike"
timeframe = "1d"
leverage = 1.0