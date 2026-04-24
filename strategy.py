#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h Donchian(20) trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for Donchian trend and squeeze detection.
- Bollinger Band Squeeze: BB Width < 20th percentile of last 50 periods indicates low volatility (6h).
- Breakout: Close > Upper BB (long) or Close < Lower BB (short) with volume > 1.5x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 12h Donchian(20) (long if price > upper, short if price < lower).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian(20) channels
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper_12h = highest_20
    donchian_lower_12h = lowest_20
    
    # Align 12h Donchian levels to 6h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    
    # Bollinger Bands (20, 2) on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Bollinger Band Squeeze: BB Width < 20th percentile of last 50 periods
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # BB width percentile + BB + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(squeeze[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Bollinger Band breakout with volume spike
            if volume_spike[i]:
                # Long breakout: close > upper BB and price > 12h Donchian upper (uptrend)
                if close[i] > upper_bb[i] and close[i] > donchian_upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < lower BB and price < 12h Donchian lower (downtrend)
                elif close[i] < lower_bb[i] and close[i] < donchian_lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Bollinger Bands or opposite signal
            if close[i] < sma_20[i]:  # Exit when price falls below middle BB
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Bollinger Bands or opposite signal
            if close[i] > sma_20[i]:  # Exit when price rises above middle BB
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BB_Squeeze_Donchian_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0