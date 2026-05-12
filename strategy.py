#!/usr/bin/env python3
# 6h_12h_1d_Anti_Martingale_Volume_Profile
# Hypothesis: Uses 12h volume profile to identify high-volume nodes (HVN) as support/resistance.
# On 6basis, enters long when price pulls back to HVN with volume confirmation and 1d trend alignment.
# Short when price rallies to HVN with volume confirmation and 1d trend alignment.
# Anti-martingale scaling: increases position size on consecutive wins, resets on loss.
# Designed for low frequency (20-50 trades/year) with strong edge in ranging markets.

name = "6h_12h_1d_Anti_Martingale_Volume_Profile"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = volumes = prices['volume'].values
    
    # Volume-weighted average price (VWAP) on 6basis for deviation
    vwap_num = (high + low + close) / 3 * volume
    vwap_den = volume
    vwap = np.nancumsum(vwap_num) / np.nancumsum(vwap_den)
    vwap = np.where(vwap_den.cumsum() == 0, np.nan, vwap)
    
    # 12h data for volume profile (high volume nodes)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h volume profile: price bins with volume
    # Use 20 bins between min and max of lookback window
    lookback = 100  # bars for profile calculation
    price_min = np.min(low[-lookback:]) if len(low) >= lookback else np.min(low)
    price_max = np.max(high[-lookback:]) if len(high) >= lookback else np.max(high)
    if price_max <= price_min:
        price_max = price_min + 1e-8
    
    bins = 20
    bin_width = (price_max - price_min) / bins
    
    # Initialize volume profile
    vol_profile = np.zeros(bins)
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    
    # For simplicity, use last 12h bar's volume distribution approximation
    # In practice, this would need historical volume distribution
    # Approximate: assume volume distributes normally around VWAP
    vwap_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vol_12h = df_12h['volume'].values
    vwap_12h_series = pd.Series(vwap_12h).ewm(span=20).mean().values
    
    # High Volume Nodes: prices where 12h volume exceeds 1.5x average
    vol_avg_12h = np.mean(vol_12h) if len(vol_12h) > 0 else 0
    hvn_threshold = 1.5 * vol_avg_12h
    
    # Simplified: use VWAP deviation as proxy for distance to HVN
    # In ranging markets, price reverts to VWAP (which approximates HVN)
    price_dev = (close - vwap) / vwap
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike on 6basis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    win_streak = 0  # consecutive wins for anti-martingale
    last_exit_price = 0
    
    for i in range(100, n):
        if (np.isnan(vwap[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            vwap[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        dev = price_dev[i]
        
        if position == 0:
            # LONG: price below VWAP (pullback to HVN) + volume spike + above 1d EMA50
            if (dev < -0.005 and  # 0.5% below VWAP
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                # Anti-martingale: increase size after wins
                size = 0.25 * (1 + min(win_streak * 0.1, 0.5))  # up to 0.375
                signals[i] = size
                position = 1
            # SHORT: price above VWAP (pullback to HVN) + volume spike + below 1d EMA50
            elif (dev > 0.005 and   # 0.5% above VWAP
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                size = 0.25 * (1 + min(win_streak * 0.1, 0.5))
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to VWAP or breaks below 1d EMA50
            if (abs(dev) < 0.002) or close[i] < ema_50_1d_aligned[i]:
                # Calculate win/loss for anti-martingale
                exit_price = close[i]
                if last_exit_price > 0:
                    if exit_price > last_exit_price:  # profit
                        win_streak += 1
                    else:
                        win_streak = 0
                last_exit_price = exit_price
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                size = 0.25 * (1 + min(win_streak * 0.1, 0.5))
                signals[i] = size
        elif position == -1:
            # EXIT SHORT: price returns to VWAP or breaks above 1d EMA50
            if (abs(dev) < 0.002) or close[i] > ema_50_1d_aligned[i]:
                exit_price = close[i]
                if last_exit_price > 0:
                    if exit_price < last_exit_price:  # profit (short)
                        win_streak += 1
                    else:
                        win_streak = 0
                last_exit_price = exit_price
                signals[i] = 0.0
                position = 0
            else:
                size = 0.25 * (1 + min(win_streak * 0.1, 0.5))
                signals[i] = -size
    
    return signals