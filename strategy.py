#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA(50) filter and 1d volume confirmation
# Long when price breaks above 4h Donchian upper band, price > 4h EMA(50), and 1d volume > 1.5x average
# Short when price breaks below 4h Donchian lower band, price < 4h EMA(50), and 1d volume > 1.5x average
# Exit when price returns to 4h EMA(50) or opposite breakout occurs
# Stoploss at 2.5 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h EMA for trend filter and 1d volume to confirm breakout strength
# Target: 75-150 total trades over 4 years (19-38/year)
# Session filter: 08-20 UTC to avoid low-volume periods

name = "1h_donchian20_4h_ema_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 4h Donchian(20) channels
    # Upper band = highest high of last 20 periods
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band = lowest low of last 20 periods
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 1d volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to EMA or breaks below lower band
            elif close[i] <= ema_4h_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to EMA or breaks above upper band
            elif close[i] >= ema_4h_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA (bullish trend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA (bearish trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals