#!/usr/bin/env python3
"""
12h_1d_1w_volume_profile_reversion_v1
Strategy: 12h volume profile mean reversion with 1d/1w trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses 12h volume profile to identify high-volume nodes (support/resistance). When price deviates significantly from the volume-weighted average price (VWAP) with confirmation from volume imbalance, mean reversion trades are taken. Trend filtered by 1d EMA50 and 1w EMA200 to avoid fighting major trends. Designed for low-frequency, high-conviction trades in ranging markets while avoiding trend exhaustion. Target: 15-35 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_volume_profile_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den > 0, vwap_num / vwap_den, typical_price)
    
    # 12h standard deviation of price from VWAP (20-period)
    price_dev = typical_price - vwap
    vwap_std = pd.Series(price_dev).rolling(window=20, min_periods=20).std().values
    
    # Volume imbalance: current volume vs 20-period average (detect absorption)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_avg > 0, vol_avg, 1.0)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        vwap_val = vwap[i]
        vwap_std_val = vwap_std[i]
        vol_ratio_val = vol_ratio[i]
        
        # Avoid division by zero or extremely small std
        if vwap_std_val <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Z-score of price deviation from VWAP
        z_score = (price_close - vwap_val) / vwap_std_val
        
        # Volume imbalance filter: look for exhaustion (low volume on extreme moves)
        vol_exhaustion = vol_ratio_val < 0.5  # Low volume on extreme move suggests exhaustion
        
        # Trend filters
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        uptrend_1w = price_close > ema_200_1w_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        downtrend_1w = price_close < ema_200_1w_aligned[i]
        
        # Mean reversion conditions
        # Long: price significantly below VWAP with volume exhaustion in uptrend context
        long_signal = (z_score < -2.0) and vol_exhaustion and uptrend_1d and uptrend_1w
        
        # Short: price significantly above VWAP with volume exhaustion in downtrend context
        short_signal = (z_score > 2.0) and vol_exhaustion and downtrend_1d and downtrend_1w
        
        # Exit when price returns to VWAP or extreme reverses
        exit_long = position == 1 and (price_close >= vwap_val or z_score > -0.5)
        exit_short = position == -1 and (price_close <= vwap_val or z_score < 0.5)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals