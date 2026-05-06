#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In strong uptrends: Bull Power > 0 and increasing, Bear Power < 0
# In strong downtrends: Bear Power < 0 and decreasing, Bull Power > 0
# Volume confirmation (>1.5x 20-bar average) ensures breakout strength
# Works in both bull/bear markets by adapting to trend direction via 1d EMA34 filter
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_ElderRay_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Elder Ray components on 6h timeframe
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate volume spike filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter_aligned[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power > 0 AND price > 1d EMA34 (uptrend) AND volume spike
            if bull_power[i] > 0 and close[i] > ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 AND price < 1d EMA34 (downtrend) AND volume spike
            elif bear_power[i] < 0 and close[i] < ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (loss of bullish momentum) OR price < 1d EMA34 (trend break)
            if bull_power[i] <= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (loss of bearish momentum) OR price > 1d EMA34 (trend break)
            if bear_power[i] >= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals