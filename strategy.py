#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume and 1d trend filter.
Enter long when price breaks above upper BB with volume > 1.5x avg and 1d close > 1d EMA50.
Enter short when price breaks below lower BB with volume > 1.5x avg and 1d close < 1d EMA50.
Exit on opposite band touch or trend reversal. Bollinger squeeze identifies low volatility
periods before explosive moves, effective in both trending and ranging markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bb_squeeze_12h_volume_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === BOLLINGER BANDS (20, 2) ===
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # === 12H VOLUME FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === 1D TREND FILTER (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ma_12h_aligned[i]) or np.isnan(daily_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bull_trend = close[i] > daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: touch lower BB or trend turns bearish
            if close[i] <= bb_lower[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: touch upper BB or trend turns bullish
            if close[i] >= bb_upper[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (12h volume > 1.5x average)
            if vol_ma_12h_aligned[i] <= 0 or volume[i] <= 1.5 * vol_ma_12h_aligned[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 1d trend
            if bull_trend:
                # In bull trend: long on break above upper BB
                if close[i] > bb_upper[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear trend: short on break below lower BB
                if close[i] < bb_lower[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals