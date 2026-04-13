#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 1d Bollinger Band breakout with 1w volume regime filter
    # Works in bull/bear: BB breakouts capture momentum, 1w volume average filter avoids low-activity chop,
    # discrete sizing (0.25) minimizes fee drag. Target: 12-25 trades/year to stay within 6h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 1w data for volume regime (average volume filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.ones(len(df_1w))
    
    # Calculate 1d Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Calculate 1w average volume (10-period)
    vol_avg_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    vol_avg_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_avg_10_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 1w volume > 1.2x 10-period average (avoid low-activity chop)
        idx_1w = i // (24 * 7)  # 1w bars in 6h timeframe (28 bars per week)
        if idx_1w >= len(volume_1w):
            signals[i] = 0.0
            continue
        volume_regime = volume_1w[idx_1w] > 1.2 * vol_avg_10_1w_aligned[i]
        
        # Entry conditions: BB breakout + volume regime
        enter_long = (close[i] > bb_upper_aligned[i]) and volume_regime
        enter_short = (close[i] < bb_lower_aligned[i]) and volume_regime
        
        # Stoploss: 1.5x ATR based on BB width (volatility-adjusted)
        bb_width = bb_upper_aligned[i] - bb_lower_aligned[i]
        stop_distance = bb_width * 0.15  # 15% of BB width
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1d_1w_bb_breakout_volume_regime_v1"
timeframe = "6h"
leverage = 1.0