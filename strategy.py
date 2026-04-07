#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(15) breakout with 1d EMA200 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper band, 1d close > 1d EMA200 (uptrend), and volume > 1.8x 12h average volume
# Short when price breaks below 12h Donchian lower band, 1d close < 1d EMA200 (downtrend), and volume > 1.8x 12h average volume
# Exit when trend reverses (1d close crosses EMA200) or opposite breakout occurs
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1d EMA200 for trend filter and 12h volume average for confirmation
# Target: 80-150 total trades over 4 years (20-38/year)

name = "12h_donchian15_1d_ema200_vol_v2"
timeframe = "12h"
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
    
    # 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian(15) channels
    high_series = pd.Series(high_12h)
    donchian_upper = high_series.rolling(window=15, min_periods=15).max().values
    low_series = pd.Series(low_12h)
    donchian_lower = low_series.rolling(window=15, min_periods=15).min().values
    
    # Align Donchian bands to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12h volume average for confirmation
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=15, min_periods=15).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_12h_aligned[i]) or 
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
            # Exit: trend reverses (price below EMA200) or breaks below lower band
            elif close[i] < ema_1d_aligned[i] or close[i] < lower_aligned[i]:
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
            # Exit: trend reverses (price above EMA200) or breaks above upper band
            elif close[i] > ema_1d_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA200 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.8 * volume_ma_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA200 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.8 * volume_ma_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals