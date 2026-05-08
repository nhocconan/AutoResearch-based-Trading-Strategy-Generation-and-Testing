#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_Trend_Follow_v1"
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
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Daily Pivot Points (previous day's HLC/3) ===
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
    
    # Pivot support/resistance levels (standard Camarilla)
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    s2 = pivot - (range_1d * 1.1 / 6)
    
    # Align pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 1d Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Trend filter: 50-period EMA ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1d ATR for volatility regime ===
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr20_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr50_1d = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr20_12h = align_htf_to_ltf(prices, df_1d, atr20_1d)
    atr50_12h = align_htf_to_ltf(prices, df_1d, atr50_1d)
    atr_ratio = atr20_12h / (atr50_12h + 1e-10)
    
    # Regime: trending if ATR ratio > 1.2, ranging if < 0.8
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(ema50_12h[i]) or np.isnan(atr_ratio[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: trending or ranging
            is_trending = atr_ratio[i] > 1.2
            is_ranging = atr_ratio[i] < 0.8
            
            if is_trending:
                # Trending regime: breakout continuation
                long_cond = (close[i] > r2_12h[i] and 
                            close[i] > ema50_12h[i] and
                            volume[i] > vol_ma20[i])
                
                short_cond = (close[i] < s2_12h[i] and 
                             close[i] < ema50_12h[i] and
                             volume[i] > vol_ma20[i])
            elif is_ranging:
                # Ranging regime: mean reversion at S1/R1
                long_cond = (close[i] < s1_12h[i] and 
                            close[i] > ema50_12h[i] and  # Avoid buying in strong downtrend
                            volume[i] > vol_ma20[i])
                
                short_cond = (close[i] > r1_12h[i] and 
                             close[i] < ema50_12h[i] and  # Avoid selling in strong uptrend
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
                # In trending market, exit on breakdown below S2 or trend reversal
                exit_cond = (close[i] < s2_12h[i] or 
                            close[i] < ema50_12h[i])
            else:
                # In ranging market, exit at R1 (mean reversion target)
                exit_cond = close[i] > r1_12h[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # In trending market, exit on breakout above R2 or trend reversal
                exit_cond = (close[i] > r2_12h[i] or 
                            close[i] > ema50_12h[i])
            else:
                # In ranging market, exit at S1 (mean reversion target)
                exit_cond = close[i] < s1_12h[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h adaptive pivot-based strategy that switches between breakout continuation
# in trending markets (detected by rising ATR) and mean reversion at S1/R1 in ranging markets.
# Uses 1d EMA50 for trend filter and volume confirmation. Designed to work in both bull (trend following) 
# and bear (mean reversion in ranges) markets. Targets 50-150 trades over 4 years (12-37/year) to minimize 
# fee drag. Uses discrete sizing (0.25) to reduce churn. Works on BTC/ETH via institutional pivot levels.