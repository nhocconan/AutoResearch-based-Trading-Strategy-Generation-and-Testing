#!/usr/bin/env python3
# Hypothesis: 1d timeframe with 1-week RSI filter and Donchian breakout for trend following.
# In trending markets (weekly RSI > 55 for long, < 45 for short), price tends to continue in the direction of the trend.
# Enters long when price breaks above daily Donchian upper (20) in bullish weekly regime,
# enters short when price breaks below daily Donchian lower (20) in bearish weekly regime.
# Exits when price crosses the daily Donchian midpoint or weekly RSI reverts to neutral zone.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_RSI_Weekly_Donchian_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly RSI (14-period) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    
    # Weekly RSI thresholds: >55 bullish, <45 bearish
    rsi_bullish = rsi_1w_values > 55
    rsi_bearish = rsi_1w_values < 45
    rsi_bullish_aligned = align_htf_to_ltf(prices, df_1w, rsi_bullish)
    rsi_bearish_aligned = align_htf_to_ltf(prices, df_1w, rsi_bearish)
    
    # Daily Donchian channel (20-period) for breakout signals
    donchian_window = 20
    high_roll = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max()
    low_roll = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Breakout conditions
    breakout_upper = close > donchian_upper
    breakout_lower = close < donchian_lower
    cross_above_mid = (close > donchian_mid) & (np.roll(close, 1) <= np.roll(donchian_mid, 1))
    cross_below_mid = (close < donchian_mid) & (np.roll(close, 1) >= np.roll(donchian_mid, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = donchian_window  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if weekly RSI data not ready
        if np.isnan(rsi_bullish_aligned[i]) or np.isnan(rsi_bearish_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish weekly RSI + price breaks above Donchian upper
            if rsi_bullish_aligned[i] and breakout_upper[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish weekly RSI + price breaks below Donchian lower
            elif rsi_bearish_aligned[i] and breakout_lower[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian mid OR weekly RSI turns bearish
            if cross_below_mid[i] or rsi_bearish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian mid OR weekly RSI turns bullish
            if cross_above_mid[i] or rsi_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals