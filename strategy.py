#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-week EMA200 trend filter and volume confirmation
# Long when price breaks above 6h Donchian upper band, weekly close > weekly EMA200 (uptrend), and volume > 2x 6h average volume
# Short when price breaks below 6h Donchian lower band, weekly close < weekly EMA200 (downtrend), and volume > 2x 6h average volume
# Exit when trend reverses (weekly close crosses EMA200) or opposite breakout occurs
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly EMA200 for trend filter and 6h volume average for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_donchian20_1w_ema200_vol_v1"
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
    
    # 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h volume average for confirmation
    volume_6h = df_6h['volume'].values
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
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
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma_6h_aligned[i]) or 
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
            # Exit: trend reverses (price below weekly EMA200) or breaks below lower band
            elif close[i] < ema_1w_aligned[i] or close[i] < lower_aligned[i]:
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
            # Exit: trend reverses (price above weekly EMA200) or breaks above upper band
            elif close[i] > ema_1w_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above weekly EMA200 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_1w_aligned[i] and
                volume[i] > 2.0 * volume_ma_6h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below weekly EMA200 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_1w_aligned[i] and
                  volume[i] > 2.0 * volume_ma_6h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals