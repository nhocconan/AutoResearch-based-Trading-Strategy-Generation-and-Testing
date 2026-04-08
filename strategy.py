#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_fractal_breakout_1d_trend_volume_v10"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA for trend filter (50-period)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR for volatility filter (14-period)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12-period high/low for fractal breakout
    high_12 = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_12 = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Calculate 12-period average volume for volume confirmation
    avg_volume_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Calculate 50-period SMA of ATR for volatility normalization
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(50, 12)  # EMA50 and 12-period lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d[i]) or np.isnan(atr_1d[i]) or 
            np.isnan(high_12[i]) or np.isnan(low_12[i]) or 
            np.isnan(avg_volume_12[i]) or np.isnan(volume[i]) or
            np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 12h bar
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)[i]
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)[i]
        atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)[i]
        
        # Volatility filter: avoid extremely low volatility (choppy) conditions
        # Use 50-period SMA of ATR to normalize
        volatility_filter = atr_1d_aligned > (atr_ma_50_aligned * 0.7)  # Only trade when volatility is above 70% of average
        
        # Volume confirmation: current volume > 2.0x average volume (stricter)
        volume_confirmation = volume[i] > (avg_volume_12[i] * 2.0)
        
        # Trend filter: price above/below 50 EMA on 1d
        uptrend = close[i] > ema_1d_aligned
        downtrend = close[i] < ema_1d_aligned
        
        if position == 1:  # Long position
            # Exit: price breaks below 12-period low OR trend reversal
            if close[i] < low_12[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 12-period high OR trend reversal
            if close[i] > high_12[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above 12-period high + uptrend + volume confirmation + volatility filter
            if close[i] > high_12[i] and uptrend and volume_confirmation and volatility_filter:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 12-period low + downtrend + volume confirmation + volatility filter
            elif close[i] < low_12[i] and downtrend and volume_confirmation and volatility_filter:
                position = -1
                signals[i] = -0.25
    
    return signals