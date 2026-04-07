#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 4h timeframe, use daily trend (close > SMA50) for direction, 
enter on Donchian(20) breakout in trend direction with volume > 1.5x average.
Exit on opposite Donchian breakout or when price crosses SMA50 (trend change).
Targets 20-50 trades/year to minimize fee drag while capturing trends.
Works in bull via breakouts, in bear via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Calculate ATR for volatility filter (if needed)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily SMA50 for trend filter
    sma50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Upper channel = max(high, 20), Lower channel = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or atr[i] <= 0 or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(sma50_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on short signal (price breaks below Donchian low with volume)
            if close[i] < donchian_low[i] and vol_confirm:
                exit_long = True
            # Exit when price crosses below SMA50 (trend change)
            elif close[i] < sma50_aligned[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on long signal (price breaks above Donchian high with volume)
            if close[i] > donchian_high[i] and vol_confirm:
                exit_short = True
            # Exit when price crosses above SMA50 (trend change)
            elif close[i] > sma50_aligned[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume + uptrend (close > SMA50)
            long_entry = (close[i] > donchian_high[i] and 
                         vol_confirm and 
                         close[i] > sma50_aligned[i])
            
            # Short entry: price breaks below Donchian low with volume + downtrend (close < SMA50)
            short_entry = (close[i] < donchian_low[i] and 
                          vol_confirm and 
                          close[i] < sma50_aligned[i])
            
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
    
    return signals