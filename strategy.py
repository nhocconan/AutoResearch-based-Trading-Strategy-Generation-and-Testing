#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and ATR stoploss
# Uses daily trend filter (price > 200-day EMA) to avoid counter-trend trades
# Target: 20-40 trades/year per symbol, low turnover, works in bull/bear via trend filter
name = "12h_Donchian20_Volume_EMA200Trend"
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
    
    # Daily trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 12h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # ATR for stoploss (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        # Trend filter: only long above EMA200, short below EMA200
        trend_filter_long = close[i] > ema200_1d_aligned[i]
        trend_filter_short = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend
            if close[i] > high_max[i] and volume_spike[i] and trend_filter_long:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and downtrend
            elif close[i] < low_min[i] and volume_spike[i] and trend_filter_short:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: break below Donchian low or trend reversal
            if close[i] < low_min[i] or not trend_filter_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: break above Donchian high or trend reversal
            if close[i] > high_max[i] or not trend_filter_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals