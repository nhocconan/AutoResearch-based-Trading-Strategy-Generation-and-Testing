#!/usr/bin/env python3

# 4h_volume_breakout_trend_v1
# Hypothesis: Volume-confirmed breakout from EMA-based channels with trend filter.
# Uses EMA(20) as center, ATR(14) for dynamic bands (2x ATR). Enters on breakout with volume > 1.5x average.
# Trend filter: price above/below daily EMA(50) to align with higher timeframe trend.
# Designed to capture momentum moves in both bull and bear markets with low trade frequency.
# Target: 20-30 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_breakout_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (daily EMA50) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h indicators
    # EMA20 for dynamic center line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for volatility bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Upper and lower bands (EMA20 ± 2*ATR)
    upper_band = ema20 + 2.0 * atr
    lower_band = ema20 - 2.0 * atr
    
    # Volume confirmation - 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(avg_volume[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: close below EMA20 or volatility contraction
            if close[i] < ema20[i] or atr[i] < atr[i-1] * 0.8:  # Volatility dropping
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: close above EMA20 or volatility contraction
            if close[i] > ema20[i] or atr[i] < atr[i-1] * 0.8:  # Volatility dropping
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Breakout conditions
            if volume_ok:
                # Long breakout: price crosses above upper band in uptrend
                if daily_uptrend and close[i] > upper_band[i] and close[i-1] <= upper_band[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price crosses below lower band in downtrend
                elif daily_downtrend and close[i] < lower_band[i] and close[i-1] >= lower_band[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals