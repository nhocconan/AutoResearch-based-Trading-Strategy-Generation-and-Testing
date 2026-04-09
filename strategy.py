#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w trend filter and volume confirmation
# Williams Fractals identify key swing highs/lows for breakout entries
# 1w EMA(34) provides higher timeframe trend bias (works in bull/bear via alignment)
# Volume > 1.5x 20-period average confirms breakout authenticity
# Discrete sizing 0.25 limits drawdown in 2022 crash (~19% loss vs 77% BTC drop)
# Target: 75-175 total trades over 4 years (19-44/year) with discrete sizing
# Novelty: Uses fractal breaks (not Donchian/Camarilla) with weekly trend filter on 6h

name = "6h_1w_fractal_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA(34)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Williams Fractals (5-bar: 2 left, center, 2 right)
    # Bearish fractal: high[i] is highest of [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest of [i-2, i-1, i, i+1, i+2]
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Bearish fractal: current high is highest in window
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = high[i]
        # Bullish fractal: current low is lowest in window
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Align fractals to 6h timeframe with 2-bar extra delay for confirmation
    # Fractals need 2 future bars to confirm, so add 2-bar delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < bullish fractal (support break) OR price < 1w EMA (trend change)
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > bearish fractal (resistance break) OR price > 1w EMA (trend change)
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and fractal break + 1w EMA filter
            if volume_confirmed:
                # Long entry: price > bearish fractal (resistance break) AND price > 1w EMA (bullish alignment)
                if close[i] > bearish_fractal_aligned[i] and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < bullish fractal (support break) AND price < 1w EMA (bearish alignment)
                elif close[i] < bullish_fractal_aligned[i] and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals