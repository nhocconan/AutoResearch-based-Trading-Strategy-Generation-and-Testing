#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Previous day's pivot points (HLC/3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Pivot support/resistance levels
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 4h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Volatility filter: 1d ATR ratio (current vs 50-period average) ===
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr20_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr50_1d = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr20_4h = align_htf_to_ltf(prices, df_1d, atr20_1d)
    atr50_4h = align_htf_to_ltf(prices, df_1d, atr50_1d)
    atr_ratio = atr20_4h / (atr50_4h + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for ATR50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(ema34_4h[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: trending if ATR ratio > 1.2, ranging if < 0.8
            is_trending = atr_ratio[i] > 1.2
            is_ranging = atr_ratio[i] < 0.8
            
            if is_trending:
                # Trending regime: breakout above R1 or below S1
                long_cond = (close[i] > r1_4h[i] and 
                            close[i] > ema34_4h[i] and
                            volume[i] > vol_ma20[i])
                
                short_cond = (close[i] < s1_4h[i] and 
                             close[i] < ema34_4h[i] and
                             volume[i] > vol_ma20[i])
            elif is_ranging:
                # Ranging regime: mean reversion at S1/R1
                long_cond = (close[i] < s1_4h[i] and 
                            close[i] > ema34_4h[i] and  # Avoid buying in strong downtrend
                            volume[i] > vol_ma20[i])
                
                short_cond = (close[i] > r1_4h[i] and 
                             close[i] < ema34_4h[i] and  # Avoid selling in strong uptrend
                             volume[i] > vol_ma20[i])
            else:
                # Transition zone: no trades
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # In trending market, exit on breakdown below S1 or trend reversal
                exit_cond = (close[i] < s1_4h[i] or 
                            close[i] < ema34_4h[i])
            else:
                # In ranging market, exit at R1 (mean reversion target)
                exit_cond = close[i] > r1_4h[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # In trending market, exit on breakout above R1 or trend reversal
                exit_cond = (close[i] > r1_4h[i] or 
                            close[i] > ema34_4h[i])
            else:
                # In ranging market, exit at S1 (mean reversion target)
                exit_cond = close[i] < s1_4h[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla-based strategy that switches between breakout continuation
# in trending markets (detected by rising ATR ratio) and mean reversion at S1/R1 in ranging
# markets. Uses 1d EMA34 for trend filter and volume confirmation. Designed to work in 
# both bull (trend following) and bear (mean reversion in ranges) markets. Targets 
# 50-150 trades over 4 years (12-37/year) to minimize fee drag. Uses discrete sizing 
# (0.25) to reduce churn. Works on BTC/ETH via institutional pivot levels.