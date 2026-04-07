#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour 1D EMA200 trend filter with 12H Donchian(20) breakout and volume confirmation
# Long when price breaks above Donchian(20) high, 1D EMA200 upward slope, and volume > 2x 12H average volume
# Short when price breaks below Donchian(20) low, 1D EMA200 downward slope, and volume > 2x 12H average volume
# Exit when price reverses to opposite Donchian boundary or EMA200 slope changes
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year) for 12H timeframe

name = "12h_donchian20_1d_ema200_vol_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12H Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1D EMA200 for trend filter (slope)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_slope = np.diff(ema200_1d, prepend=ema200_1d[0])
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema200_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d_slope)
    
    # 12H volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(ema200_1d_slope_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: price breaks below Donchian low or EMA200 slope turns negative
            elif close[i] < donchian_low[i] or ema200_1d_slope_aligned[i] < 0:
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
            # Exit: price breaks above Donchian high or EMA200 slope turns positive
            elif close[i] > donchian_high[i] or ema200_1d_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, EMA200 slope alignment, and volume confirmation
            # Bullish breakout: price breaks above Donchian(20) high
            bullish_break = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]
            # Bearish breakout: price breaks below Donchian(20) low
            bearish_break = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]
            
            # Long: bullish breakout, EMA200 upward slope, volume spike
            if (bullish_break and
                ema200_1d_slope_aligned[i] > 0 and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, EMA200 downward slope, volume spike
            elif (bearish_break and
                  ema200_1d_slope_aligned[i] < 0 and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals