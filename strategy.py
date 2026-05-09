#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d timeframe for structure and trend.
# Uses 1-day KAMA direction (adaptive trend) combined with 12h price action
# relative to 1-day Bollinger Bands (20, 2) for mean-reversion entries in ranging markets
# and trend-following exits. Includes volume confirmation and chop regime filter.
# Designed to work in both bull and bear markets by adapting to market regime.
# Target: 15-25 trades per year to minimize fee drag and improve generalization.
name = "12h_KAMA_BBands_Chop_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA, Bollinger Bands, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day KAMA (adaptive trend)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i < 10:  # not enough data for ER
            er[i] = 0
        else:
            change_val = np.abs(close_1d[i] - close_1d[i-9])
            volatility_val = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = change_val / volatility_val if volatility_val != 0 else 0
    # Smoothing constants
    sc = (er * 0.59 + 0.06) ** 2  # 2/(2+1) - 2/(30+1) => fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1-day Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate 1-day Chop Index (14) for regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum.reduce([tr1.values, tr2.values, tr3.values])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    upper_bb_12h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_12h = align_htf_to_ltf(prices, df_1d, lower_bb)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: spike above 1.5x 24-period average (2 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Wait for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_12h[i]) or np.isnan(upper_bb_12h[i]) or 
            np.isnan(lower_bb_12h[i]) or np.isnan(chop_12h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        # Determine market regime: chop > 61.8 = ranging, chop < 38.2 = trending
        is_ranging = chop_12h[i] > 61.8
        is_trending = chop_12h[i] < 38.2
        
        if position == 0:
            # In ranging markets: mean reversion at Bollinger Bands
            if is_ranging:
                # Long: price at lower BB with volume confirmation
                if close[i] <= lower_bb_12h[i] and vol_ok and in_session:
                    signals[i] = 0.25
                    position = 1
                # Short: price at upper BB with volume confirmation
                elif close[i] >= upper_bb_12h[i] and vol_ok and in_session:
                    signals[i] = -0.25
                    position = -1
            # In trending markets: follow KAMA direction
            elif is_trending:
                # Long: price above KAMA (uptrend) with volume confirmation
                if close[i] > kama_12h[i] and vol_ok and in_session:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA (downtrend) with volume confirmation
                elif close[i] < kama_12h[i] and vol_ok and in_session:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            exit_signal = False
            if is_ranging:
                # Exit long when price reaches middle of BB or upper BB
                if close[i] >= sma_20_12h[i] if 'sma_20_12h' in locals() else close[i] >= (upper_bb_12h[i] + lower_bb_12h[i])/2:
                    exit_signal = True
            else:  # trending
                # Exit long when price crosses below KAMA
                if close[i] < kama_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            exit_signal = False
            if is_ranging:
                # Exit short when price reaches middle of BB or lower BB
                if close[i] <= sma_20_12h[i] if 'sma_20_12h' in locals() else close[i] <= (upper_bb_12h[i] + lower_bb_12h[i])/2:
                    exit_signal = True
            else:  # trending
                # Exit short when price crosses above KAMA
                if close[i] > kama_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Pre-calculate SMA 20 for 1d to use in exit conditions (avoid recomputation in loop)
# This is a simplification - in practice we'd align it like other indicators
# For now, we'll calculate it inside the loop where needed using available data
# Note: The above code has been adjusted to handle the SMA reference properly
# by calculating it when needed or using a simplified mid-point.