#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_Volume_ATRStop_v1
Hypothesis: Trade 4h Donchian(20) breakouts with 1d EMA50 trend filter, volume confirmation (1.5x median), and ATR(14) stoploss.
Donchian channels provide clear structure for breakouts. HTF trend ensures we trade with the higher timeframe momentum.
Volume confirms institutional interest. ATR stop manages risk. Designed for low trade frequency (20-40/year) on 4h to minimize fee drag.
Uses discrete position sizing (0.30) to reduce churn. Works in bull/bear markets by following 1d EMA50 trend.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Warmup: max of Donchian(20), EMA50(1d), ATR(14), volume median(20)
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend (close > 1d EMA50)
            long_signal = (close_val > donchian_high_val) and \
                          (volume_val > 1.5 * vol_median_val) and \
                          (close_val > ema_50_1d_val)
            
            # Short: break below Donchian low with volume and downtrend (close < 1d EMA50)
            short_signal = (close_val < donchian_low_val) and \
                           (volume_val > 1.5 * vol_median_val) and \
                           (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                atr_at_entry = atr_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: ATR-based stoploss or trend reversal
            if close_val < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Optional: exit on Donchian low break (mean reversion)
            elif close_val < donchian_low_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: ATR-based stoploss or trend reversal
            if close_val > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Optional: exit on Donchian high break (mean reversion)
            elif close_val > donchian_high_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0