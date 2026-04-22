#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 12h EMA trend filter and volume spike
# Long when price breaks above bearish fractal in uptrend (close > 12h EMA50) with volume spike (>2x 20-period avg)
# Short when price breaks below bullish fractal in downtrend (close < 12h EMA50) with volume spike
# Exit when price retouches the last opposite fractal or trend reverses
# Williams Fractals identify potential reversal points; breakouts from these levels with volume and trend confirmation
# capture strong moves. Designed for low trade frequency (~20-40/year) to minimize fee drain.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: high[n-2] > high[n-1] < high[n] and high[n] < high[n+1] and high[n] < high[n+2]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        if (high_1d[i-2] > high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i] < high_1d[i+1] and 
            high_1d[i] < high_1d[i+2]):
            bullish_fractal[i] = high_1d[i]
    
    # Align fractals to 4h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        ema_val = ema_50_12h_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above bearish fractal + uptrend + volume spike
            if price > bear_fractal and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below bullish fractal + downtrend + volume spike
            elif price < bull_fractal and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price retouches opposite fractal or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price touches or crosses bullish fractal or trend turns down
                if price <= bull_fractal or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price touches or crosses bearish fractal or trend turns up
                if price >= bear_fractal or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0