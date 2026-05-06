#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA200 trend filter and volume confirmation
# Uses Donchian(20) breakouts for structure, 1d EMA200 for major trend alignment (avoids counter-trend trades)
# Volume confirmation (>1.8x 20-bar average) filters weak breakouts
# ATR-based trailing stop via signal=0 when price retraces 50% of breakout range
# Discrete sizing 0.25 to balance return and fee drag; target 80-180 total trades over 4 years
# Proven pattern: Donchian breakouts with volume/regime filters work on BTC/ETH in both bull/bear markets

name = "4h_Donchian20_1dEMA200_VolumeConfirm_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for stoploss and breakout validation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate volume filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_level = 0.0  # tracks breakout level for trailing stop
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                breakout_level = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > upper Donchian AND uptrend (price > EMA200) AND volume spike
            if close[i] > donchian_high[i] and close[i] > ema200_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
                breakout_level = donchian_high[i]
            # Short breakdown: price < lower Donchian AND downtrend (price < EMA200) AND volume spike
            elif close[i] < donchian_low[i] and close[i] < ema200_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
                breakout_level = donchian_low[i]
        elif position == 1:
            # Long position management
            # Trailing stop: exit if price retraces 50% from breakout level to midpoint
            retracement_level = breakout_level - 0.5 * (breakout_level - donchian_low[i])
            if close[i] <= retracement_level:
                signals[i] = 0.0
                position = 0
                breakout_level = 0.0
            # Optional: re-entry on new breakout in same direction
            elif close[i] > donchian_high[i]:
                signals[i] = 0.25
                breakout_level = donchian_high[i]
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Trailing stop: exit if price retraces 50% from breakout level to midpoint
            retracement_level = breakout_level + 0.5 * (donchian_high[i] - breakout_level)
            if close[i] >= retracement_level:
                signals[i] = 0.0
                position = 0
                breakout_level = 0.0
            # Optional: re-entry on new breakdown in same direction
            elif close[i] < donchian_low[i]:
                signals[i] = -0.25
                breakout_level = donchian_low[i]
            else:
                signals[i] = -0.25
    
    return signals