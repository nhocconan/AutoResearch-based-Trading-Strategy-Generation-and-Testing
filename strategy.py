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
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(21) for trend
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Get daily data for entry signals and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian(20) channels
    high_20_1d = np.full(len(df_1d), np.nan)
    low_20_1d = np.full(len(df_1d), np.nan)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    high_20_1d[19:] = high_series.rolling(window=20, min_periods=20).max().values[19:]
    low_20_1d[19:] = low_series.rolling(window=20, min_periods=20).min().values[19:]
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate daily volume moving average
    vol_s_1d = pd.Series(volume_1d)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    vol_ma_20_1d[19:] = vol_s_1d.rolling(window=20, min_periods=20).mean().values[19:]
    
    # Align daily indicators to 12h timeframe
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR 20-period MA for volatility filter
    atr_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(33, len(df_1d)):  # 14 + 19 for 20-period MA
        atr_ma_20_1d[i] = np.mean(atr_1d[i-19:i+1])
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(high_20_1d_aligned[i]) or np.isnan(low_20_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period daily volume MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Volatility filter: daily ATR > 0.3 * its 20-period MA (avoid low volatility)
        vol_filter_volatility = atr_1d_aligned[i] > 0.3 * atr_ma_20_1d_aligned[i]
        
        # Trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_1d_aligned[i]
        short_breakout = close[i] < low_20_1d_aligned[i]
        
        # Entry conditions: breakout in trend direction + volume + volatility filter
        long_entry = long_breakout and uptrend and vol_filter and vol_filter_volatility
        short_entry = short_breakout and downtrend and vol_filter and vol_filter_volatility
        
        # Exit conditions: opposite breakout or volatility drop
        long_exit = close[i] < low_20_1d_aligned[i]
        short_exit = close[i] > high_20_1d_aligned[i]
        
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

name = "12h_1w_1d_donchian_breakout_trend_vol_filter_v2"
timeframe = "12h"
leverage = 1.0