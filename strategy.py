#!/usr/bin/env python3
# 6h_1d_OrderBlock_Trend_Follow_v1
# Hypothesis: On 6h timeframe, trade breakouts from 1d order blocks with trend following.
# In bull/bear markets, price respects institutional order blocks (accumulation/distribution zones).
# Uses 1d volume profile to identify high-volume nodes (HVNs) as support/resistance.
# Trend filter: 1d EMA50 - only trade in direction of higher timeframe trend.
# Targets 20-40 trades/year by requiring confluence of order block, volume, and trend.

name = "6h_1d_OrderBlock_Trend_Follow_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Identify 1d order blocks (high volume nodes)
    # For each day, if volume > 1.5x 20-day average, mark the close as significant level
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * volume_ma_20)
    
    # Create order block levels: use close of high volume days
    ob_levels = np.where(volume_spike, close_1d, np.nan)
    
    # Forward fill to create persistent levels until next significant volume day
    ob_series = pd.Series(ob_levels)
    ob_filled = ob_series.ffill().bfill().values  # Fill both directions for robustness
    
    # Alternative: use volume-weighted average price for significant days
    vwap_1d_num = (high_1d + low_1d + close_1d) * volume_1d  # Typical price * volume
    vwap_1d_den = volume_1d
    vwap_1d = np.where(vwap_1d_den > 0, vwap_1d_num / vwap_1d_den, 0)
    
    # Use VWAP of high volume days as stronger support/resistance
    vwap_ob = np.where(volume_spike, vwap_1d, np.nan)
    vwap_ob_series = pd.Series(vwap_ob)
    vwap_ob_filled = vwap_ob_series.ffill().bfill().values
    
    # Align to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ob_level_aligned = align_htf_to_ltf(prices, df_1d, ob_filled)
    vwap_ob_aligned = align_htf_to_ltf(prices, df_1d, vwap_ob_filled)
    
    # Volume average for spike detection on 6h
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(ob_level_aligned[i]) or 
            np.isnan(vwap_ob_aligned[i]) or np.isnan(volume_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakouts from order block levels in direction of 1d trend
            # Uptrend: price > EMA50, look for long breakouts above OB/VWAP
            # Downtrend: price < EMA50, look for short breakdowns below OB/VWAP
            
            if close[i] > ema50_aligned[i]:  # Uptrend filter
                # Long breakout above order block or VWAP with volume
                ob_level = ob_level_aligned[i]
                vwap_level = vwap_ob_aligned[i]
                
                # Use the higher of OB or VWAP as resistance in uptrend
                resistance = max(ob_level, vwap_level) if not (np.isnan(ob_level) or np.isnan(vwap_level)) else \
                            (ob_level if not np.isnan(ob_level) else vwap_level)
                
                if not np.isnan(resistance):
                    if (close[i] > resistance * 1.002 and  # 0.2% breakout
                        volume[i] > 1.5 * volume_ma_6h[i]):
                        signals[i] = 0.25
                        position = 1
                        
            elif close[i] < ema50_aligned[i]:  # Downtrend filter
                # Short breakdown below order block or VWAP with volume
                ob_level = ob_level_aligned[i]
                vwap_level = vwap_ob_aligned[i]
                
                # Use the lower of OB or VWAP as support in downtrend
                support = min(ob_level, vwap_level) if not (np.isnan(ob_level) or np.isnan(vwap_level)) else \
                         (ob_level if not np.isnan(ob_level) else vwap_level)
                
                if not np.isnan(support):
                    if (close[i] < support * 0.998 and  # 0.2% breakdown
                        volume[i] > 1.5 * volume_ma_6h[i]):
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit: trend reversal or return to order block
            if close[i] < ema50_aligned[i] or close[i] < ob_level_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: trend reversal or return to order block
            if close[i] > ema50_aligned[i] or close[i] > ob_level_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals