#!/usr/bin/env python3
"""
1h_StructureMomentum_WithTrendFilter_V1
1h momentum with 4h/1d trend and structure filter:
- Long: price > 4h EMA50 AND price > 1d EMA50 AND close > open (bullish candle)
- Short: price < 4h EMA50 AND price < 1d EMA50 AND close < open (bearish candle)
- Exit when trend alignment breaks
- Volume filter: volume > 1.2x 20-period average
- Designed for 15-35 trades/year per symbol
Works in bull (follow 4h/1d uptrend) and bear (follow 4h/1d downtrend)
"""

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
    open_price = prices['open'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need 50 for EMA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.2x 20-period average
        volume_ok = volume[i] > 1.2 * vol_ma[i]
        
        # Trend alignment
        above_both_emas = close[i] > ema_50_4h_aligned[i] and close[i] > ema_50_1d_aligned[i]
        below_both_emas = close[i] < ema_50_4h_aligned[i] and close[i] < ema_50_1d_aligned[i]
        
        # Candle direction
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Long: above both EMAs + bullish candle + volume
            if above_both_emas and bullish_candle and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: below both EMAs + bearish candle + volume
            elif below_both_emas and bearish_candle and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend alignment breaks or bearish candle with volume
            if not above_both_emas or (bearish_candle and volume_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend alignment breaks or bullish candle with volume
            if not below_both_emas or (bullish_candle and volume_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_StructureMomentum_WithTrendFilter_V1"
timeframe = "1h"
leverage = 1.0