#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v1
# Hypothesis: Williams Fractals on 1d timeframe identify potential turning points.
# A bullish fractal break above the prior day's high with volume confirmation and
# daily EMA trend alignment provides high-probability long entries. Conversely,
# a bearish fractal break below the prior day's low with volume and trend alignment
# provides short entries. The 1d EMA acts as a dynamic trend filter, ensuring trades
# align with the higher timeframe direction. This strategy aims for low trade frequency
# (target: 15-40 trades/year) by requiring multiple confluence factors, reducing
# fee drag and improving robustness in both bull and bear markets.

name = "4h_fractal_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: highest high with two lower highs on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: lowest low with two higher lows on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
            
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d
    bearish_fractal, bullish_fractal = calculate_williams_fractals(
        df_1d['high'].values, df_1d['low'].values
    )
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe with proper delay for fractals
    # Fractals need 2 additional bars for confirmation (total 3 bars: 1 forming + 2 confirming)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h volume ratio (current vs 20-period average) for confirmation
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # Track position state
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after sufficient warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Close below EMA(50) OR bearish fractal break with volume
            if (close[i] < ema_50_aligned[i] or 
                (low[i] < bearish_aligned[i] and vol_ratio[i] > 1.5)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Close above EMA(50) OR bullish fractal break with volume
            if (close[i] > ema_50_aligned[i] or 
                (high[i] > bullish_aligned[i] and vol_ratio[i] > 1.5)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Bullish fractal break above prior day's high with volume and uptrend
            bull_break = (high[i] > bullish_aligned[i] and 
                         vol_ratio[i] > 1.5 and
                         close[i] > ema_50_aligned[i])
            
            # Short: Bearish fractal break below prior day's low with volume and downtrend
            bear_break = (low[i] < bearish_aligned[i] and 
                         vol_ratio[i] > 1.5 and
                         close[i] < ema_50_aligned[i])
            
            if bull_break:
                position = 1
                signals[i] = 0.25
            elif bear_break:
                position = -1
                signals[i] = -0.25
    
    return signals