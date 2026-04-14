#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day RSI(14) and Williams %R(14) mean reversion.
# In overbought conditions (RSI > 70 and Williams %R > -20), go short.
# In oversold conditions (RSI < 30 and Williams %R < -80), go long.
# Volume > 1.5x 20-period average confirms reversal strength.
# Uses 4h ATR(14) for stoploss: exit when price moves 2x ATR against position.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by fading extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    
    # Load 4h data ONCE for ATR stoploss
    df_4h = get_htf_data(prices, '4h')
    
    # 1d RSI(14)
    rsi_len = 14
    if len(df_1d) < rsi_len:
        return np.zeros(n)
    
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with df_1d index
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1d Williams %R(14)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=rsi_len, min_periods=rsi_len).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=rsi_len, min_periods=rsi_len).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    williams_r = np.concatenate([[np.nan], williams_r])  # align with df_1d index
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 4h ATR(14) for stoploss
    atr_len = 14
    if len(df_4h) < atr_len:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(rsi_len*2, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion conditions
        oversold = (rsi_aligned[i] < 30) and (williams_r_aligned[i] < -80)
        overbought = (rsi_aligned[i] > 70) and (williams_r_aligned[i] > -20)
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if oversold and volume_confirmed:
                position = 1
                signals[i] = position_size
            elif overbought and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price drops 2x ATR from entry or mean reversion unwinds
            # Track entry price approximation using close at signal bar
            if i > 0:
                # Approximate entry: use close of previous bar (signal acts on close, fills next open)
                approx_entry = close[i-1] if i > 0 else close[i]
                stop_loss = approx_entry - 2 * atr_aligned[i]
                
                if close[i] < stop_loss or rsi_aligned[i] > 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises 2x ATR from entry or mean reversion unwinds
            if i > 0:
                approx_entry = close[i-1] if i > 0 else close[i]
                stop_loss = approx_entry + 2 * atr_aligned[i]
                
                if close[i] > stop_loss or rsi_aligned[i] < 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI_Williams_MeanRev_v1"
timeframe = "4h"
leverage = 1.0