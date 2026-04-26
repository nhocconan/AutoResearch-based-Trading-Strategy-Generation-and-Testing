#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion
Hypothesis: Williams Vix Fix (WVF) identifies extreme fear/greed on 6b timeframe. 
Long when WVF > 0.8 (extreme fear) and price < 6h VWAP (oversold + below fair value).
Short when WVF < 0.2 (extreme greed) and price > 6h VWAP (overbought + above fair value).
Exit when WVF returns to neutral range (0.4-0.6) or price crosses VWAP.
Works in bull/bear markets by fading extremes while respecting intraday value area.
Uses 1d trend filter to avoid counter-trend extremes in strong moves.
Target: 15-25 trades/year (60-100 total over 4 years).
"""

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
    volume = prices['volume'].values
    
    # Williams Vix Fix: measures market fear (0-1, >0.8 = extreme fear)
    # WVF = ((Highest Close in Period - Low) / (Highest Close in Period)) * 100
    # We normalize to 0-1 range
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    wvf = ((highest_close - low) / highest_close) * 100
    wvf = wvf / 100  # normalize to 0-1
    
    # 6h VWAP (volume weighted average price)
    typical_price = (high + low + close) / 3
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_denominator = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_numerator / vwap_denominator
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter (avoid counter-trend extremes)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (22 for WVF, 20 for VWAP, 50 for 1d EMA)
    start_idx = max(22, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wvf[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        wvf_val = wvf[i]
        vwap_val = vwap[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        # Determine 1d trend
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: extreme WVF + price deviation from VWAP, aligned with 1d trend
        # Long: extreme fear (WVF > 0.8) + price below VWAP (oversold) in bullish 1d regime
        long_entry = (wvf_val > 0.8) and (close_val < vwap_val) and bullish_1d
        
        # Short: extreme greed (WVF < 0.2) + price above VWAP (overbought) in bearish 1d regime
        short_entry = (wvf_val < 0.2) and (close_val > vwap_val) and bearish_1d
        
        # Exit conditions: WVF returns to neutral OR price crosses VWAP
        wvf_neutral = (wvf_val >= 0.4) and (wvf_val <= 0.6)
        price_at_vwap = abs(close_val - vwap_val) < (vwap_val * 0.001)  # within 0.1% of VWAP
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion
            if wvf_neutral or price_at_vwap or not bullish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion
            if wvf_neutral or price_at_vwap or not bearish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion"
timeframe = "6h"
leverage = 1.0