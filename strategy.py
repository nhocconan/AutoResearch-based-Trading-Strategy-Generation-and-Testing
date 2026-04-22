#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA trend filter and volume confirmation
# Long when price breaks above bearish fractal resistance in uptrend (price > 12h EMA50)
# Short when price breaks below bullish fractal support in downtrend (price < 12h EMA50)
# Uses volume spike (2x 20-period average) to confirm breakouts
# Williams Fractals identified on 12h timeframe for better structure
# Target: 15-25 trades/year per symbol, works in bull/bear via trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for Williams Fractals and EMA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Fractals on 12h data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    n_12h = len(high_12h)
    bearish_fractal = np.full(n_12h, np.nan)
    bullish_fractal = np.full(n_12h, np.nan)
    
    for i in range(2, n_12h - 2):
        # Bearish fractal (peak)
        if (high_12h[i-2] < high_12h[i-1] and 
            high_12h[i] > high_12h[i-1] and 
            high_12h[i] > high_12h[i+1] and 
            high_12h[i+1] > high_12h[i+2]):
            bearish_fractal[i] = high_12h[i]
        
        # Bullish fractal (trough)
        if (low_12h[i-2] > low_12h[i-1] and 
            low_12h[i] < low_12h[i-1] and 
            low_12h[i] < low_12h[i+1] and 
            low_12h[i+1] < low_12h[i+2]):
            bullish_fractal[i] = low_12h[i]
    
    # Calculate 50-period EMA on 12h close for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50 = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 6-hour timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above bearish fractal resistance + volume spike + uptrend (price > EMA50)
            if (close[i] > bearish_fractal_aligned[i] and vol_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bullish fractal support + volume spike + downtrend (price < EMA50)
            elif (close[i] < bullish_fractal_aligned[i] and vol_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite fractal level
            if position == 1:
                if close[i] < bullish_fractal_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > bearish_fractal_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_Trend_Volume_Session"
timeframe = "6h"
leverage = 1.0