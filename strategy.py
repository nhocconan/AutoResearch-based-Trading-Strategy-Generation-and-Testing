#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-period high + weekly close > weekly open (bullish weekly) + volume > 1.5x 20-period average volume
# Short when price breaks below 20-period low + weekly close < weekly open (bearish weekly) + volume > 1.5x 20-period average volume
# Exit when price crosses opposite Donchian boundary or weekly trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly trend filter to avoid counter-trend trades and volume confirmation to avoid false breakouts
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1w_trend_vol_v1"
timeframe = "12h"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly bullish/bearish: close > open = bullish, close < open = bearish
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish weekly candle
    
    # Align weekly trend to 12h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # 12-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 20-period average volume for volume confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit: price breaks below Donchian low or weekly trend turns bearish
            elif close[i] < donchian_low[i] or weekly_bullish_aligned[i] <= 0.5:
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
            # Exit: price breaks above Donchian high or weekly trend turns bullish
            elif close[i] > donchian_high[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and weekly trend filter
            volume_confirmation = volume[i] > 1.5 * avg_volume[i]
            
            # Long: price breaks above Donchian high + weekly bullish + volume confirmation
            if close[i] > donchian_high[i] and weekly_bullish_aligned[i] > 0.5 and volume_confirmation:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + weekly bearish + volume confirmation
            elif close[i] < donchian_low[i] and weekly_bullish_aligned[i] <= 0.5 and volume_confirmation:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals