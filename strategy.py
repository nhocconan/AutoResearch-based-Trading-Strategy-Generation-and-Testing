#!/usr/bin/env python3
"""
6h_1d_1w_OrderFlow_Imbalance
Hypothesis: Combines volume-weighted price action with multi-timeframe structure to identify institutional order flow imbalances.
Uses 1w trend bias, 1d volume profile value area, and 6s delta imbalance to capture smart money entries.
Works in bull/bear markets by following higher timeframe structure while entering on lower timeframe imbalances.
Target: 15-35 trades/year on 6f (60-140 total over 4 years).
"""

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
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # Calculate buy/sell volume delta
    sell_volume = volume - taker_buy_volume
    delta = taker_buy_volume - sell_volume  # positive = buying pressure
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly trend: higher highs and higher lows = bullish, lower highs and lower lows = bearish
    weekly_higher_high = weekly_high >= np.roll(weekly_high, 1)
    weekly_higher_low = weekly_low >= np.roll(weekly_low, 1)
    weekly_lower_high = weekly_high <= np.roll(weekly_high, 1)
    weekly_lower_low = weekly_low <= np.roll(weekly_low, 1)
    
    weekly_bullish = weekly_higher_high & weekly_higher_low
    weekly_bearish = weekly_lower_high & weekly_lower_low
    
    # Get daily data for value area (volume profile)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily value area (simplified: VWAP +- 1 ATR)
    typical_price = (daily_high + daily_low + daily_close) / 3
    vwap = np.cumsum(typical_price * daily_volume) / np.cumsum(daily_volume)
    atr = np.zeros_like(daily_close)
    for i in range(1, len(daily_close)):
        tr = max(
            daily_high[i] - daily_low[i],
            abs(daily_high[i] - daily_close[i-1]),
            abs(daily_low[i] - daily_close[i-1])
        )
        atr[i] = 0.99 * atr[i-1] + 0.01 * tr if i > 1 else tr
    
    va_high = vwap + 0.5 * atr
    va_low = vwap - 0.5 * atr
    
    # Align all data to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    va_high_aligned = align_htf_to_ltf(prices, df_1d, va_high)
    va_low_aligned = align_htf_to_ltf(prices, df_1d, va_low)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Volume imbalance: current delta vs 20-period average
    delta_ma = pd.Series(delta).rolling(window=20, min_periods=20).mean()
    delta_std = pd.Series(delta).rolling(window=20, min_periods=20).std()
    delta_zscore = np.where(delta_std > 0, (delta - delta_ma) / delta_std, 0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(va_high_aligned[i]) or np.isnan(va_low_aligned[i]) or
            np.isnan(vwap_aligned[i]) or np.isnan(delta_zscore[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Long setup: weekly bullish bias + price below value area + strong buying imbalance
        long_setup = (
            weekly_bullish_aligned[i] > 0.5 and
            price < va_low_aligned[i] and
            delta_zscore[i] > 1.5
        )
        
        # Short setup: weekly bearish bias + price above value area + strong selling imbalance
        short_setup = (
            weekly_bearish_aligned[i] > 0.5 and
            price > va_high_aligned[i] and
            delta_zscore[i] < -1.5
        )
        
        # Exit conditions: price returns to VWAP or imbalance dissipates
        long_exit = (
            position == 1 and
            (price > vwap_aligned[i] or delta_zscore[i] < 0.5)
        )
        
        short_exit = (
            position == -1 and
            (price < vwap_aligned[i] or delta_zscore[i] > -0.5)
        )
        
        if long_setup and position != 1:
            position = 1
            signals[i] = position_size
        elif short_setup and position != -1:
            position = -1
            signals[i] = -position_size
        elif long_exit:
            position = 0
            signals[i] = 0.0
        elif short_exit:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_OrderFlow_Imbalance"
timeframe = "6h"
leverage = 1.0