#!/usr/bin/env python3
# 12h_keltner_breakout_daily_trend_volume_v1
# Hypothesis: 12h Keltner channel breakout with daily trend filter and volume confirmation. 
# Uses EMA(20) as center line and ATR(10) for channel width (1.5x ATR). 
# Enters long when price breaks above upper band in daily uptrend with volume surge.
# Enters short when price breaks below lower band in daily downtrend with volume surge.
# Exits when price crosses back below/above EMA(20) or volatility contracts.
# Designed for low trade frequency (15-25/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_keltner_breakout_daily_trend_volume_v1"
timeframe = "12h"
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
    
    # Daily trend filter (EMA50) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h indicators
    # EMA20 for center line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for channel width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels (EMA20 ± 1.5*ATR)
    upper_band = ema20 + 1.5 * atr
    lower_band = ema20 - 1.5 * atr
    
    # Volume confirmation (20-period average)
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