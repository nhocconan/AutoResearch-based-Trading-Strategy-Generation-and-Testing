#!/usr/bin/env python3
# Hypothesis: 1-day Bollinger Band width contraction (volatility squeeze) followed by expansion with price breaking above/below upper/lower band, confirmed by weekly trend via 50-week SMA and volume spike. Designed for low trade frequency (<25/year) to capture explosive moves after consolidation in both bull and bear markets.

name = "1d_BollingerSqueeze_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Width Squeeze: width < 50th percentile of past 50 days
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.5).values
    squeeze = bb_width < bb_width_percentile
    
    # Weekly trend filter: 50-week SMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    sma50_1w = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Volume filter: current volume > 20-day average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bollinger squeeze breakout above upper band with weekly uptrend and volume
            if squeeze[i-1] and close[i] > bb_upper[i] and close[i] > sma50_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bollinger squeeze breakout below lower band with weekly downtrend and volume
            elif squeeze[i-1] and close[i] < bb_lower[i] and close[i] < sma50_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back below middle band
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back above middle band
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals