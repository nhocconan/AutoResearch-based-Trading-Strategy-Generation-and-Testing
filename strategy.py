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
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily Bollinger Bands (20, 2.0)
    close_1d_series = pd.Series(df_1d['close'].values)
    ma20 = close_1d_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_1d_series.rolling(window=20, min_periods=20).std().values
    upper_band = ma20 + 2.0 * std20
    lower_band = ma20 - 2.0 * std20
    
    # Align Bollinger Bands to 6h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Weekly trend filter: price above/below weekly EMA20
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Bollinger Band width for volatility regime
    bb_width = (upper_band - lower_band) / ma20
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(bb_width_aligned[i]) or 
            np.isnan(vol_ma[i])):
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
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Volatility filter: low volatility regime (BB width below median)
        vol_filter_low = bb_width_aligned[i] < np.nanmedian(bb_width_aligned[:i+1])
        
        # Entry conditions: 
        # Long: price touches lower Bollinger Band in uptrend + low volatility
        # Short: price touches upper Bollinger Band in downtrend + low volatility
        long_entry = (close[i] <= lower_band_aligned[i]) and vol_filter and trend_up and vol_filter_low
        short_entry = (close[i] >= upper_band_aligned[i]) and vol_filter and trend_down and vol_filter_low
        
        # Exit conditions: price returns to middle Bollinger Band
        middle_band_aligned = align_htf_to_ltf(prices, df_1d, ma20)
        long_exit = (close[i] >= middle_band_aligned[i]) and position == 1
        short_exit = (close[i] <= middle_band_aligned[i]) and position == -1
        
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

name = "6h_BollingerMeanReversion_WeeklyTrend_Volume_Session"
timeframe = "6h"
leverage = 1.0