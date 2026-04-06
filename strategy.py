#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1-day EMA200 trend filter and volume confirmation
# Long when price breaks above 6h Donchian upper band, price > 1d EMA200 (bullish bias), and volume > 1.3x 60-period average
# Short when price breaks below 6h Donchian lower band, price < 1d EMA200 (bearish bias), and volume > 1.3x 60-period average
# Exit when opposite breakout occurs or price crosses back through EMA200
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses 1d EMA200 for trend bias and volume confirmation to avoid false breakouts
# Target: 80-150 total trades over 4 years (20-38/year)

name = "6h_donchian20_1d_ema200_vol_v1"
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
    
    # 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # 6h Donchian(20) channels
    high_series = pd.Series(high_6h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_6h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # 1d EMA200 for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Volume confirmation: 60-period average
    volume_ma = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    # ATR(20) for stoploss
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
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(volume_ma[i]) or 
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
            # Exit: price crosses below EMA200 or breaks below lower band
            elif close[i] < ema200_aligned[i] or close[i] < lower_aligned[i]:
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
            # Exit: price crosses above EMA200 or breaks above upper band
            elif close[i] > ema200_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend bias
            # Long: price breaks above upper band, price > EMA200 (bullish), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema200_aligned[i] and
                volume[i] > 1.3 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price < EMA200 (bearish), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema200_aligned[i] and
                  volume[i] > 1.3 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals