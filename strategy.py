#!/usr/bin/env python3
"""
4h_1d_WilsonBreakout_Volume_Trend_v1
Concept: Wilson Cycle Breakout with Volume Confirmation and Trend Filter on 4h timeframe.
- Uses Wilson Cycle concept: Price breaking above/below recent high/low with volume confirmation
- Long when price breaks above 20-period high with volume > 1.5x average and price above 50-period EMA
- Short when price breaks below 20-period low with volume > 1.5x average and price below 50-period EMA
- Exit when price returns to 10-period EMA (mean reversion)
- Uses daily trend filter: Only take long when daily close > 200 EMA, short when daily close < 200 EMA
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: Trend filter adapts to market conditions, volume confirms breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_WilsonBreakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === Daily: 200 EMA trend filter ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h: Price channel (20-period high/low) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 20-period high and low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h: EMA50 trend filter ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 4h: EMA10 for exit (mean reversion target) ===
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema200_1d_val = ema200_1d_aligned[i]
        ema50_val = ema50[i]
        ema10_val = ema10[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema200_1d_val) or np.isnan(ema50_val) or np.isnan(ema10_val) or 
            np.isnan(high_20_val) or np.isnan(low_20_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-period high with volume confirmation
            # AND price above 50 EMA AND daily trend bullish (close > 200 EMA)
            breakout_high = close_val > high_20_val
            vol_confirm = vol_ratio_val > 1.5
            trend_4h_bull = close_val > ema50_val
            trend_1d_bull = close_1d[i//16] > ema200_1d[i//16] if i//16 < len(ema200_1d) else False  # Daily close > 200 EMA
            
            if breakout_high and vol_confirm and trend_4h_bull and trend_1d_bull:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low with volume confirmation
            # AND price below 50 EMA AND daily trend bearish (close < 200 EMA)
            elif (close_val < low_20_val and vol_confirm and 
                  ema50_val > close_val and 
                  close_1d[i//16] < ema200_1d[i//16] if i//16 < len(ema200_1d) else False):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to 10-period EMA (mean reversion)
            if close_val <= ema10_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to 10-period EMA (mean reversion)
            if close_val >= ema10_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals