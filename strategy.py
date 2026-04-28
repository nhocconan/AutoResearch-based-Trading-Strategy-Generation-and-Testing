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
    
    # Get weekly data for ATR-based regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly ATR(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR is inf
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get daily data for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian Channel (20)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        # Regime filter: low volatility (ATR below median)
        # Use 50-period median of ATR to avoid look-ahead
        if i >= 50:
            atr_median = np.median(atr_14_aligned[i-50:i])
            low_vol = atr_14_aligned[i] < atr_median
        else:
            low_vol = True  # Not enough data for median, allow trading
        
        # Entry conditions: 
        # Long: breakout above daily Donchian high in uptrend + low vol
        # Short: breakdown below daily Donchian low in downtrend + low vol
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up and low_vol
        short_entry = short_breakout and vol_filter and trend_down and low_vol
        
        # Exit conditions: opposite Donchian level touch
        long_exit = (close[i] < low_20_aligned[i]) and position == 1
        short_exit = (close[i] > high_20_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_DailyTrend_Volume_LowVol_Session"
timeframe = "1d"
leverage = 1.0