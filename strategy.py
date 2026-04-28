#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate ATR-based volatility regime (high volatility = trending)
    atr_ma_14 = pd.Series(atr_1d_aligned).rolling(window=14, min_periods=14).mean().values
    vol_regime = atr_1d_aligned > atr_ma_14  # High volatility regime
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA21 trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_ma_14[i])):
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
        
        # Trend filter: price above/below weekly EMA21
        trend_up = close[i] > ema21_1w_aligned[i]
        trend_down = close[i] < ema21_1w_aligned[i]
        
        # Volatility regime filter: only trade in high volatility (trending) regime
        vol_filter_regime = vol_regime[i]
        
        # Entry conditions: 
        # Long: price above weekly EMA21 with volume and high volatility regime
        # Short: price below weekly EMA21 with volume and high volatility regime
        long_entry = trend_up and vol_filter and vol_filter_regime
        short_entry = trend_down and vol_filter and vol_filter_regime
        
        # Exit conditions: trend reversal or low volatility regime
        long_exit = (not trend_up) or (not vol_filter_regime)
        short_exit = (not trend_down) or (not vol_filter_regime)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "4h_WeeklyEMA21_Trend_Volume_VolatilityRegime"
timeframe = "4h"
leverage = 1.0