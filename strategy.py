#!/usr/bin/env python3

"""
Hypothesis: 4-hour Bollinger Band Squeeze with 1-day ADX trend filter and volume confirmation.
Trades breakouts from low volatility squeeze (BB width < 20th percentile) in the direction of the daily ADX trend.
Uses volume spike to confirm institutional interest at breakout. Designed for low trade frequency
(15-30 trades/year) to minimize fee drift and work in both bull and bear markets by combining
volatility contraction breakouts with trend alignment.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands upper, lower, and width."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    width = (upper - lower) / sma  # Normalized width
    return upper.values, lower.values, width.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    tr = np.zeros_like(high)
    for i in range(len(high)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for ADX trend filter and Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Daily Bollinger Bands for squeeze detection (20-period, 2 std)
    bb_upper_1d, bb_lower_1d, bb_width_1d = calculate_bollinger_bands(close_1d, period=20, std_dev=2.0)
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Calculate 20th percentile of BB width for squeeze threshold (using expanding window)
    bb_width_percentile = np.zeros_like(bb_width_1d_aligned)
    for i in range(len(bb_width_1d_aligned)):
        if i < 20:
            bb_width_percentile[i] = np.nan
        else:
            bb_width_percentile[i] = np.percentile(bb_width_1d_aligned[:i+1], 20)
    
    # 4-hour Bollinger Bands for breakout detection
    bb_upper_4h, bb_lower_4h, _ = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(bb_width_1d_aligned[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB width below 20th percentile (low volatility)
        squeeze = bb_width_1d_aligned[i] < bb_width_percentile[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and squeeze and vol_spike:
            # Long: price breaks above upper BB with uptrend bias (ADX > 25)
            if close[i] > bb_upper_4h[i] and adx_14_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB with downtrend bias (ADX > 25)
            elif close[i] < bb_lower_4h[i] and adx_14_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of BB or volatility expands (squeeze ends)
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below middle BB or squeeze ends
                bb_middle_4h = (bb_upper_4h[i] + bb_lower_4h[i]) / 2
                if close[i] < bb_middle_4h or not squeeze:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price closes above middle BB or squeeze ends
                bb_middle_4h = (bb_upper_4h[i] + bb_lower_4h[i]) / 2
                if close[i] > bb_middle_4h or not squeeze:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Bollinger_Squeeze_ADX14_Volume"
timeframe = "4h"
leverage = 1.0