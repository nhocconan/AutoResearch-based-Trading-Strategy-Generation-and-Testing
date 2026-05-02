#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend (bull/bear regime) and 6h Williams fractals for precise reversal entries
# Entry: Long when bullish fractal forms above 6h EMA20 with volume spike and price > 1d EMA34 (uptrend)
#        Short when bearish fractal forms below 6h EMA20 with volume spike and price < 1d EMA34 (downtrend)
# Exit: Close crosses 6h EMA20 (trend change) or opposite fractal level broken
# Williams fractals provide high-probability reversal points; EMA34 filters for primary trend direction
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_WilliamsFractal_Breakout_1dEMA34_Volume"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h EMA20 for dynamic support/resistance
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h Williams Fractals
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = high[i]
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bearish_fractal[i]) or np.isnan(bullish_fractal[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish fractal forms above 6h EMA20 AND price > 1d EMA34 (uptrend) AND volume spike
            if (not np.isnan(bullish_fractal[i]) and 
                bullish_fractal[i] > ema_20[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish fractal forms below 6h EMA20 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (not np.isnan(bearish_fractal[i]) and 
                  bearish_fractal[i] < ema_20[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 6h EMA20 (trend change) OR bearish fractal breaks below 6h EMA20
            if (close[i] < ema_20[i] or 
                not np.isnan(bearish_fractal[i]) and bearish_fractal[i] < ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 6h EMA20 (trend change) OR bullish fractal breaks above 6h EMA20
            if (close[i] > ema_20[i] or 
                not np.isnan(bullish_fractal[i]) and bullish_fractal[i] > ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals