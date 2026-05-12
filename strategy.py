# 1d_AroonOscillator_WeeklyTrend_VolumeSpike
# Hypothesis: Aroon Oscillator (25) identifies strong trends (|Aroon|>70) with weekly trend filter (price > weekly EMA20) and volume confirmation (>2x 20-period average).
# Works in bull markets (strong uptrends) and bear markets (strong downtrends). Uses daily timeframe with weekly trend filter to avoid counter-trend trades.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.

name = "1d_AroonOscillator_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Aroon Oscillator (25 period) - measures trend strength
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low in the lookback period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        # Find periods since highest high and lowest low
        if highest_high == high[i]:
            periods_since_high = 0
        else:
            # Look backward from i-1 to find highest high
            lookback_high = high[i - period + 1:i]
            periods_since_high = period - 1 - np.argmax(lookback_high[::-1]) if len(lookback_high) > 0 else period - 1
            
        if lowest_low == low[i]:
            periods_since_low = 0
        else:
            # Look backward from i-1 to find lowest low
            lookback_low = low[i - period + 1:i]
            periods_since_low = period - 1 - np.argmin(lookback_low[::-1]) if len(lookback_low) > 0 else period - 1
            
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    aroon_osc = aroon_up - aroon_down  # Range: -100 to +100

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(25, n):
        if np.isnan(aroon_osc[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Strong uptrend (Aroon > 70) + price above weekly EMA20 + volume spike
            if aroon_osc[i] > 70 and close[i] > ema20_1w_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = 0.25
                position = 1
            # SHORT: Strong downtrend (Aroon < -70) + price below weekly EMA20 + volume spike
            elif aroon_osc[i] < -70 and close[i] < ema20_1w_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakens (Aroon < 30) or price breaks below weekly EMA20
            if aroon_osc[i] < 30 or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens (Aroon > -30) or price breaks above weekly EMA20
            if aroon_osc[i] > -30 or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals