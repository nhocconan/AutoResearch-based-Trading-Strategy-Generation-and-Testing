#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_BollingerBandBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d Bollinger Bands (20, 2) for volatility breakout
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # 1d volume filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1w trend filter: price above/below 50-period EMA
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or \
           np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_ok = vol > 1.5 * vol_ma
        
        # Bollinger Band breakout conditions
        breakout_up = price > upper_bb_val
        breakout_down = price < lower_bb_val
        
        if position == 0:
            # Long: upward breakout + volume + above weekly EMA50
            if breakout_up and volume_ok and price > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + below weekly EMA50
            elif breakout_down and volume_ok and price < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to middle of Bollinger Bands or breaks lower band
            if price < sma_20[-1] if len(sma_20) > 0 else False or price < lower_bb_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to middle of Bollinger Bands or breaks upper band
            if price > sma_20[-1] if len(sma_20) > 0 else False or price > upper_bb_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals