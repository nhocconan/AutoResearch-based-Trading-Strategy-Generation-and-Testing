#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price action filtered by weekly trend and volume confirmation
# Long when price closes above daily 20-period high with volume > 1.5x average and weekly trend up
# Short when price closes below daily 20-period low with volume > 1.5x average and weekly trend down
# Exit on opposite 5-day close crossover
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses daily price/volume and weekly trend filter
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag

name = "1d_price_action_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for price action and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily 20-period high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily 20-period volume average for confirmation
    volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend: compare current close to close 5 periods ago
    close_1w = df_1w['close'].values
    weekly_trend = np.where(close_1w >= np.roll(close_1w, 5), 1, -1)
    weekly_trend[0:5] = 0  # Not enough data
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # ATR(20) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_20[i]) or np.isnan(weekly_trend_aligned[i]) or 
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
            # Exit: close below 5-day low
            elif close[i] < low_20[i]:
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
            # Exit: close above 5-day high
            elif close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price breakout with volume confirmation and weekly trend
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_20[i]
            
            # Long: price breaks above daily 20-period high + volume + weekly uptrend
            if close[i] > high_20[i] and volume_filter and weekly_trend_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below daily 20-period low + volume + weekly downtrend
            elif close[i] < low_20[i] and volume_filter and weekly_trend_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals