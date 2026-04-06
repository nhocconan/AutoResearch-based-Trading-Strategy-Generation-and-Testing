#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation.
# Goes long when price breaks above 4h Donchian(20) high with 1h volume > 1.5x average.
# Goes short when price breaks below 4h Donchian(20) low with 1h volume > 1.5x average.
# Uses 1d EMA trend filter to align with higher timeframe trend.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled risk.
# Session filter: 08-20 UTC to avoid low-volume Asian session.

name = "1h_donchian20_4h_trend_vol_v1"
timeframe = "1h"
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
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels (20-period high/low)
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (shifted by 1 bar for completed 4h bars)
    donch_high = align_htf_to_ltf(prices, df_4h, high_roll)
    donch_low = align_htf_to_ltf(prices, df_4h, low_roll)
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for breakouts
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below 4h Donchian low or 1d EMA turns down
            elif close[i] < donch_low[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above 4h Donchian high or 1d EMA turns up
            elif close[i] > donch_high[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation
            if vol_strong[i]:
                # Long breakout: price breaks above 4h Donchian high with strong volume
                if close[i] > donch_high[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below 4h Donchian low with strong volume
                elif close[i] < donch_low[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals