#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when price breaks above 20-period high AND 1d EMA50 uptrend AND volume > 1.5x average
# Short when price breaks below 20-period low AND 1d EMA50 downtrend AND volume > 1.5x average
# Uses ATR(14) for dynamic position sizing: 0.30 when ATR < 5% of price, else 0.15
# Designed for 12-30 trades/year per symbol to minimize fee drag while capturing major trends
# Works in both bull and bear markets by following higher timeframe trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period) on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 50-period EMA on daily close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ATR(14) for dynamic sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Dynamic position size based on volatility
    atr_ratio = np.where(close > 0, atr / close, 0)
    position_size = np.where(atr_ratio < 0.05, 0.30, 0.15)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(position_size[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above Donchian high AND 1d uptrend AND volume
        if (close[i] > high_max[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = position_size[i]
            position = 1
        # Short conditions: price breaks below Donchian low AND 1d downtrend AND volume
        elif (close[i] < low_min[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -position_size[i]
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = position_size[i]
            elif position == -1:
                signals[i] = -position_size[i]
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_DonchianBreakout_1dTrend_Volume_DynamicSize"
timeframe = "12h"
leverage = 1.0