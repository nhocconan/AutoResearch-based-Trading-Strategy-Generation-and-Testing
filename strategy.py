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
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(10) for long-term trend
    close_1w_series = pd.Series(close_1w)
    ema_10_1w = close_1w_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate daily Donchian(20) channels
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1_d = np.abs(high_1d - low_1d)
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_d[0] = tr2_d[0] = tr3_d[0] = np.nan
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_d[i-14:i+1])
    
    # Calculate daily volume moving average
    vol_s_1d = pd.Series(volume_1d)
    vol_ma_20_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 1d timeframe (no alignment needed for same timeframe)
    high_20_1d_aligned = high_20_1d
    low_20_1d_aligned = low_20_1d
    atr_1d_aligned = atr_1d
    vol_ma_20_1d_aligned = vol_ma_20_1d
    
    # Align weekly EMA to 1d timeframe
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(high_20_1d_aligned[i]) or 
            np.isnan(low_20_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period daily volume MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Volatility filter: daily ATR > 0.3 * its 20-period MA (avoid low volatility)
        atr_ma_20_1d = np.full(len(df_1d), np.nan)
        for j in range(34, len(df_1d)):  # 14 + 19 for 20-period MA
            if not np.isnan(np.mean(atr_1d[j-19:j+1])):
                atr_ma_20_1d[j] = np.mean(atr_1d[j-19:j+1])
        atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
        vol_filter_volatility = (not np.isnan(atr_ma_20_1d_aligned[i]) and 
                                atr_1d_aligned[i] > 0.3 * atr_ma_20_1d_aligned[i])
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_1d_aligned[i]
        short_breakout = close[i] < low_20_1d_aligned[i]
        
        # Trend filter: weekly EMA10 direction
        weekly_uptrend = close[i] > ema_10_1w_aligned[i]
        weekly_downtrend = close[i] < ema_10_1w_aligned[i]
        
        # Entry conditions: breakout + trend + volume + volatility filter
        long_entry = long_breakout and weekly_uptrend and vol_filter and vol_filter_volatility
        short_entry = short_breakout and weekly_downtrend and vol_filter and vol_filter_volatility
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = (close[i] < low_20_1d_aligned[i]) or (not weekly_uptrend)
        short_exit = (close[i] > high_20_1d_aligned[i]) or (not weekly_downtrend)
        
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

name = "1d_1w_donchian_breakout_ema10_vol_filter_v1"
timeframe = "1d"
leverage = 1.0