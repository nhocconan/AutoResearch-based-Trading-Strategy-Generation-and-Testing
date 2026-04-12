#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(200)
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate 12h ATR(14) for position sizing
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_12h = np.full(n, np.nan)
    for i in range(14, n):
        atr_12h[i] = np.mean(tr_h[i-14:i+1])
    
    # Align daily indicators to 12h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h volume moving average
    vol_s = pd.Series(volume)
    vol_ma_20_12h = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.5 * 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_12h[i]
        
        # Volatility filter: daily ATR > 0.5 * its 20-period MA (avoid low volatility)
        atr_ma_20_1d = np.full(len(df_1d), np.nan)
        for j in range(34, len(df_1d)):  # 14 + 19 for 20-period MA
            if not np.isnan(np.mean(atr_1d[j-19:j+1])):
                atr_ma_20_1d[j] = np.mean(atr_1d[j-19:j+1])
        atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
        vol_filter_daily = (not np.isnan(atr_ma_20_1d_aligned[i]) and 
                           atr_1d_aligned[i] > 0.5 * atr_ma_20_1d_aligned[i])
        
        # Trend filter: price above/below daily EMA200
        above_ema = close[i] > ema_200_1d_aligned[i]
        below_ema = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions: trend + volume + volatility filter
        long_entry = above_ema and vol_filter and vol_filter_daily
        short_entry = below_ema and vol_filter and vol_filter_daily
        
        # Exit conditions: trend reversal
        long_exit = below_ema
        short_exit = above_ema
        
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

name = "12h_1d_ema200_vol_vol_filter_v1"
timeframe = "12h"
leverage = 1.0