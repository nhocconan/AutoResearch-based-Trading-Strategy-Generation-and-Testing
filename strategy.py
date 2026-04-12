#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian channel (20)
    donch_high_20 = np.full(len(df_1d), np.nan)
    donch_low_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donch_high_20[i] = np.max(high_1d[i-19:i+1])
        donch_low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate daily ATR(14) for volatility
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Align daily indicators to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h ATR(14) for position sizing
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_12h = np.full(n, np.nan)
    for i in range(14, n):
        atr_12h[i] = np.mean(tr_h[i-14:i+1])
    
    # Calculate 12h volume moving average
    vol_s_h = pd.Series(volume)
    vol_ma_20_h = vol_s_h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily ATR moving average (20) for volatility filter
    atr_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(33, len(df_1d)):  # 14 + 19 for 20-period MA
        atr_ma_20_1d[i] = np.mean(atr_1d[i-19:i+1])
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma_20_h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.5 * 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_h[i]
        
        # Volatility filter: daily ATR > 0.5 * its 20-period MA (avoid low volatility)
        vol_filter_daily = atr_1d_aligned[i] > 0.5 * atr_ma_20_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > donch_high_20_aligned[i]
        breakout_short = close[i] < donch_low_20_aligned[i]
        
        # Entry conditions: breakout with volume and volatility filters
        long_entry = breakout_long and vol_filter and vol_filter_daily
        short_entry = breakout_short and vol_filter and vol_filter_daily
        
        # Exit conditions: price crosses back to opposite Donchian band
        long_exit = close[i] < donch_low_20_aligned[i]
        short_exit = close[i] > donch_high_20_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_vol_filter_v1"
timeframe = "12h"
leverage = 1.0