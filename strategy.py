#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1-week pivot direction and volume confirmation
# Long when price breaks above 20-period high, weekly pivot shows bullish bias (close > pivot), and volume > 1.5x average
# Short when price breaks below 20-period low, weekly pivot shows bearish bias (close < pivot), and volume > 1.5x average
# Exit when price returns to 10-period midpoint or trend reverses
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses weekly pivot for trend bias and Donchian for breakout signals
# Target: 50-150 total trades over 4 years (12-38/year)

name = "6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 10-period midpoint for exit
    donchian_mid_10 = (high_series.rolling(window=10, min_periods=10).max() + 
                       low_series.rolling(window=10, min_periods=10).min()) / 2
    donchian_mid_10 = donchian_mid_10.values
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6s timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Weekly bias: bullish if close > pivot, bearish if close < pivot
    weekly_bullish = close_1w > pivot_1w
    weekly_bearish = close_1w < pivot_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # 6s volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to 10-period midpoint or weekly bias turns bearish
            elif close[i] <= donchian_mid_10[i] or weekly_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to 10-period midpoint or weekly bias turns bullish
            elif close[i] >= donchian_mid_10[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with weekly bias and volume confirmation
            # Bullish breakout: price breaks above Donchian high
            bullish_breakout = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]
            # Bearish breakout: price breaks below Donchian low
            bearish_breakout = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]
            
            # Long: bullish breakout, weekly bullish bias, volume spike
            if (bullish_breakout and
                weekly_bullish_aligned[i] > 0.5 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, weekly bearish bias, volume spike
            elif (bearish_breakout and
                  weekly_bearish_aligned[i] > 0.5 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals