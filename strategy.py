#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend alignment to avoid counter-trend trades, reducing whipsaw in ranging markets
# Volume spike (>1.8x 20-bar average) confirms breakout strength
# Donchian breakout provides clear structure with proven edge on BTC/ETH in both bull/bear regimes
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Discrete sizing 0.25 to balance return and drawdown; target 60-120 total trades over 4 years (15-30/year)
# Proven pattern: price channel breakouts with volume confirmation and trend filter work on major cryptos

name = "6h_Donchian20_1dEMA34_VolumeConfirm_v1"
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
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian(20) channels (using previous 20 periods)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume spike filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper Donchian AND uptrend (price > EMA34) AND volume spike
            if close[i] > high_max_20[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower Donchian AND downtrend (price < EMA34) AND volume spike
            elif close[i] < low_min_20[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests lower Donchian from above (trend reversal)
            if close[i] <= low_min_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests upper Donchian from below (trend reversal)
            if close[i] >= high_max_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals