#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h/1d trend filters and volume confirmation
# Long when price breaks above 4h Donchian high, 1d EMA200 uptrend, and volume > 1.5x average
# Short when price breaks below 4h Donchian low, 1d EMA200 downtrend, and volume > 1.5x average
# Uses 4h Donchian(20) for structure, 1d EMA200 for trend filter, and volume to filter false breakouts
# Session filter (08-20 UTC) to reduce noise. Target: 60-150 total trades over 4 years.
# ATR-based stoploss to limit drawdown (1.5x ATR)

name = "1h_donchian20_4h_1d_ema200_vol_v1"
timeframe = "1h"
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    # 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) - using previous completed bar (shifted by align function)
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # EMA200 calculation
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[close[0]], close[:-1]]) - high)
    tr3 = np.abs(np.concatenate([[close[0]], close[:-1]]) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if required data not available or outside session
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 1.5 * ATR
            if close[i] < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below 4h Donchian low or trend changes
            elif close[i] < donch_low_4h_aligned[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 1.5 * ATR
            if close[i] > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above 4h Donchian high or trend changes
            elif close[i] > donch_high_4h_aligned[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and session filter
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            if np.isnan(vol_ma[i]):
                signals[i] = 0.0
                continue
                
            # Long: price breaks above 4h Donchian high, uptrend, volume spike
            if (close[i] > donch_high_4h_aligned[i] and 
                close[i] > ema200_1d_aligned[i] and
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below 4h Donchian low, downtrend, volume spike
            elif (close[i] < donch_low_4h_aligned[i] and 
                  close[i] < ema200_1d_aligned[i] and
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals