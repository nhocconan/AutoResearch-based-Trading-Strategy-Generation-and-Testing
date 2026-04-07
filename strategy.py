#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA200 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band, 12h close > 12h EMA200 (uptrend), and volume > 1.5x 4h average volume
# Short when price breaks below 4h Donchian lower band, 12h close < 12h EMA200 (downtrend), and volume > 1.5x 4h average volume
# Exit when trend reverses (12h close crosses EMA200) or opposite breakout occurs
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 12h EMA200 for trend filter and 4h volume average for confirmation
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_12h_ema200_vol_v1"
timeframe = "4h"
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
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=200, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h volume average for confirmation
    volume_4h = df_4h['volume'].values
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
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
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma_4h_aligned[i]) or 
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
            elif close[i] < ema_12h_aligned[i] or close[i] < lower_aligned[i]:
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
            elif close[i] > ema_12h_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA200 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_12h_aligned[i] and
                volume[i] > 1.5 * volume_ma_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA200 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_12h_aligned[i] and
                  volume[i] > 1.5 * volume_ma_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals