#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day high + 1w close > 1w open (bullish weekly) + volume > 1.5x 20-day avg volume
# Short when price breaks below 20-day low + 1w close < 1w open (bearish weekly) + volume > 1.5x 20-day avg volume
# Exit when price returns to 10-day midpoint or opposite breakout occurs
# Stoploss at 2.0 * ATR(10)
# Position size: 0.25 (25% of capital)
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1-week trend: bullish if close > open, bearish if close < open
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # 20-day Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-day midpoint for exit
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    midpoint_10 = (high_10 + low_10) / 2
    
    # 20-day average volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 10-day ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to 10-day midpoint or opposite breakout
            elif close[i] <= midpoint_10[i] or low[i] <= low_min[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to 10-day midpoint or opposite breakout
            elif close[i] >= midpoint_10[i] or high[i] >= high_max[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with weekly trend and volume confirmation
            vol_surge = volume[i] > 1.5 * vol_ma[i]
            
            # Long: break above 20-day high + weekly bullish + volume surge
            if high[i] > high_max[i] and weekly_bullish_aligned[i] > 0.5 and vol_surge:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below 20-day low + weekly bearish + volume surge
            elif low[i] < low_min[i] and weekly_bearish_aligned[i] > 0.5 and vol_surge:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals