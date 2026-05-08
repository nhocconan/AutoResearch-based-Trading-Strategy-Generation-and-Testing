#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Williams_Fractal_Breakout_1dTrend_Volume"
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
    
    # Daily data for Williams Fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (bearish = sell signal, bullish = buy signal)
    # Bearish fractal: high[n-2] < high[n-1] and high[n] < high[n-1] and high[n+1] < high[n-1] and high[n+2] < high[n-1]
    # Bullish fractal: low[n-2] > low[n-1] and low[n] > low[n-1] and low[n+1] > low[n-1] and low[n+2] > low[n-1]
    bearish_fractal = np.zeros(len(high_1d))
    bullish_fractal = np.zeros(len(low_1d))
    
    for i in range(2, len(high_1d) - 2):
        if (high[i-2] < high[i-1] and high[i] < high[i-1] and 
            high[i+1] < high[i-1] and high[i+2] < high[i-1]):
            bearish_fractal[i] = high[i-1]
        if (low[i-2] > low[i-1] and low[i] > low[i-1] and 
            low[i+1] > low[i-1] and low[i+2] > low[i-1]):
            bullish_fractal[i] = low[i-1]
    
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 after)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA34 for trend filter (only needs completed 1d candle)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above bullish fractal resistance, daily uptrend, volume spike
            long_cond = (close[i] > bullish_fractal_aligned[i] and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below bearish fractal support, daily downtrend, volume spike
            short_cond = (close[i] < bearish_fractal_aligned[i] and 
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below bearish fractal support
            if close[i] < bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above bullish fractal resistance
            if close[i] > bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals