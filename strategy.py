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
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily timeframe
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-period SMA on daily timeframe for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # Calculate 12h price range (high-low) for volatility regime
    price_range_12h = high - low
    range_ma20 = pd.Series(price_range_12h).rolling(window=20, min_periods=20).mean().values
    
    # Volatility regime: current range > 1.5x 20-period average range AND ATR > 0
    vol_regime = (price_range_12h > 1.5 * range_ma20) & (atr_14_aligned > 0)
    
    # Trend filter: price above/below daily SMA50
    uptrend = close > sma_50_aligned
    downtrend = close < sma_50_aligned
    
    # Breakout conditions: price breaks 12-period high/low on 12h chart
    high_12 = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_12 = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    breakout_up = close > high_12
    breakout_down = close < low_12
    
    # Volume filter: current volume > 1.5x 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(sma_50_aligned[i]) or
            np.isnan(range_ma20[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions: breakout + volatility regime + volume filter + trend alignment
        long_entry = breakout_up[i] and vol_regime[i] and vol_filter[i] and uptrend[i]
        short_entry = breakout_down[i] and vol_regime[i] and vol_filter[i] and downtrend[i]
        
        # Exit conditions: opposite breakout or loss of trend
        long_exit = breakout_down[i] or not uptrend[i]
        short_exit = breakout_up[i] or not downtrend[i]
        
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_VolRegime_Breakout_SMA50_Trend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0