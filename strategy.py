#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation
# Uses 4h Williams Fractals for structure breakout signals (proven edge from top performers)
# 1d EMA50 for trend filter (long-term trend alignment)
# Volume spike (2.0x 20-period average) for confirmation
# Entry logic: Long when price breaks above recent 4h bearish fractal high AND price > 1d EMA50 AND volume spike
#              Short when price breaks below recent 4h bullish fractal low AND price < 1d EMA50 AND volume spike
# Exit logic: Exit when price crosses the 1d EMA50 (trend reversal) or opposite fractal level
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Discrete sizing 0.20 balances profit potential and fee drag

name = "1h_WilliamsFractal_Breakout_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Williams Fractals (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Williams Fractals: bearish (sell) fractal = high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    #                 bullish (buy) fractal = low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.full(len(high_4h), np.nan)
    bullish_fractal = np.full(len(low_4h), np.nan)
    
    for i in range(2, len(high_4h) - 2):
        if (high_4h[i-2] < high_4h[i-1] and 
            high_4h[i] < high_4h[i-1] and 
            high_4h[i-1] > high_4h[i-3] and 
            high_4h[i-1] > high_4h[i+1]):
            bearish_fractal[i-1] = high_4h[i-1]
        
        if (low_4h[i-2] > low_4h[i-1] and 
            low_4h[i] > low_4h[i-1] and 
            low_4h[i-1] < low_4h[i-3] and 
            low_4h[i-1] < low_4h[i+1]):
            bullish_fractal[i-1] = low_4h[i-1]
    
    # Align Williams Fractals to 1h timeframe with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above recent 4h bearish fractal high AND price > 1d EMA50 (uptrend) AND volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: Break below recent 4h bullish fractal low AND price < 1d EMA50 (downtrend) AND volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA50 (trend change) OR break below recent 4h bullish fractal low (reversal)
            if (close[i] < ema_50_1d_aligned[i] or 
                close[i] < bullish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA50 (trend change) OR break above recent 4h bearish fractal high (reversal)
            if (close[i] > ema_50_1d_aligned[i] or 
                close[i] > bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals