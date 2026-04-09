#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Fractal breakout with volume confirmation
# Williams Fractals identify key swing highs/lows that act as support/resistance
# Break above recent bearish fractal (sell signal invalidated) or break below recent bullish fractal (buy signal invalidated)
# Volume confirmation ensures breakout validity
# Works in both bull/bear markets: fractals adapt to price structure, volume filters false breakouts
# Target: 12-37 trades/year on 6h timeframe

name = "6h_1d_williams_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Fractals on 1d
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal (sell signal)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]
        
        # Bullish fractal (buy signal)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]
    
    # Calculate 1d EMA(34) for trend filter
    close_s_1d = pd.Series(close_1d)
    ema_34_1d = close_s_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d average volume for confirmation
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    # Williams fractals need extra 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Pre-compute 6h volume MA for confirmation
    vol_s_6h = pd.Series(volume)
    vol_ma_20_6h = vol_s_6h.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average 1d volume
        volume_confirmed = volume[i] > 1.3 * avg_vol_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if price falls below recent bullish fractal or breaks EMA down
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_34_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above recent bearish fractal or breaks EMA up
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_34_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout logic: enter on fractal break with volume confirmation and trend alignment
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_34_aligned[i] and  # Only long in uptrend
                volume_confirmed):
                position = 1
                signals[i] = 0.25
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_34_aligned[i] and  # Only short in downtrend
                  volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals