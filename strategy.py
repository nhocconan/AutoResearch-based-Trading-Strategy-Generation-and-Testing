#!/usr/bin/env python3
"""
4h_KAMA_Trend_VolumeSpike_ChopFilter_v1
Hypothesis: KAMA adapts to market noise — in trending markets it follows price closely, in choppy markets it flattens. 
Combined with volume spike (>2x average) to confirm institutional interest and Choppiness Index > 61.8 to ensure we are in a ranging regime (mean-reversion friendly). 
Enter long when price crosses above KAMA with volume and chop filter, short when price crosses below KAMA with volume and chop filter.
Exit when price re-crosses KAMA (mean reversion completion). Designed for 4h to target 20-50 trades/year with discrete sizing (0.25).
Works in both bull and bear regimes because chop filter ensures we only trade in ranging markets where mean reversion is valid, and volume spike confirms momentum behind the move.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Pad volatility to match length
    volatility = np.concatenate([np.full(er_len-1, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constant
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # seed
    for i in range(er_len+1, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Average volume for confirmation (24-period SMA = 4h * 6 = 24h)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]  # first TR
    for i in range(1, len(close)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA seed, volume, chop
    start_idx = max(er_len + 5, 24, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        kama_val = kama[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        chop_val = chop[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(avg_vol) or np.isnan(chop_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        # Chop filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_val > 61.8
        
        # Long: price crosses above KAMA with volume and chop filter
        long_condition = (close_val > kama_val) and (close[i-1] <= kama[i-1]) and volume_confirmed and chop_filter
        # Short: price crosses below KAMA with volume and chop filter
        short_condition = (close_val < kama_val) and (close[i-1] >= kama[i-1]) and volume_confirmed and chop_filter
        
        # Exit: price re-crosses KAMA (mean reversion completion)
        long_exit = (position == 1 and close_val <= kama_val)
        short_exit = (position == -1 and close_val >= kama_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_KAMA_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0