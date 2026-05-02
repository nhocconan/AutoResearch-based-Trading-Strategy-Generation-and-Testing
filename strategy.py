#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with Weekly EMA34 Trend Filter and Volume Confirmation
# Uses 1w Williams fractals (confirmed with 2-bar delay) for breakout signals
# 1w EMA34 for trend filter (long when price > EMA34, short when price < EMA34)
# Volume confirmation: 1.8x 20-period average
# Works in both bull and bear markets by trading with the 1w trend
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_WilliamsFractal_Breakout_1wEMA34_Volume"
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w Williams Fractals (requires 2-bar confirmation delay)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Fractal: bearish = high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    #             bullish  = low[n]  < low[n-2]  and low[n]  < low[n-1]  and low[n]  < low[n+1]  and low[n]  < low[n+2]
    n_1w = len(high_1w)
    bearish_fractal = np.full(n_1w, np.nan)
    bullish_fractal = np.full(n_1w, np.nan)
    
    for i in range(2, n_1w - 2):
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]  # resistance level
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]   # support level
    
    # Align fractals to 6h timeframe with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above bullish fractal AND price > 1w EMA34 (uptrend) AND volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below bearish fractal AND price < 1w EMA34 (downtrend) AND volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1w EMA34 (trend change) OR break below nearest bullish fractal (support)
            if (close[i] < ema_34_1w_aligned[i] or 
                close[i] < bullish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1w EMA34 (trend change) OR break above nearest bearish fractal (resistance)
            if (close[i] > ema_34_1w_aligned[i] or 
                close[i] > bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals