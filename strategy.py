#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
# Long when: price breaks above recent bearish fractal (resistance) AND 1d EMA(34) rising AND volume > 1.5x 20-period average
# Short when: price breaks below recent bullish fractal (support) AND 1d EMA(34) falling AND volume > 1.5x 20-period average
# Exit when price returns to the opposite fractal level or volume drops below average.
# Designed for 6h timeframe with moderate trade frequency (target: 20-50/year) to avoid fee drag.
# Uses 1d for trend direction and volume confirmation to avoid false breakouts in choppy markets.
# Works in bull markets via upside breakouts in uptrend, in bear markets via downside breakouts in downtrend.
# Volume filter ensures breakouts have conviction, reducing false signals.
name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
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
    
    # Williams Fractals (5-bar window: highest high in center, lowest low in center)
    # Bearish fractal: highest high with lower highs on both sides
    # Bullish fractal: lowest low with higher lows on both sides
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        if (high[i] >= high[i-1] and high[i] >= high[i-2] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = high[i]
        if (low[i] <= low[i-1] and low[i] <= low[i-2] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Forward fill fractal levels to use as support/resistance until broken
    bearish_fractal_filled = pd.Series(bearish_fractal).ffill().values
    bullish_fractal_filled = pd.Series(bullish_fractal).ffill().values
    
    # 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_filled[i]) or np.isnan(bullish_fractal_filled[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) AND 1d EMA34 rising AND volume confirmed
            long_condition = (close[i] > bearish_fractal_filled[i]) and ema_34_rising_aligned[i] and volume_confirmed[i]
            # Short: price breaks below bullish fractal (support) AND 1d EMA34 falling AND volume confirmed
            short_condition = (close[i] < bullish_fractal_filled[i]) and ema_34_falling_aligned[i] and volume_confirmed[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to bullish fractal (support) OR volume drops below average
            if (close[i] < bullish_fractal_filled[i]) or (volume[i] < vol_ma_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to bearish fractal (resistance) OR volume drops below average
            if (close[i] > bearish_fractal_filled[i]) or (volume[i] < vol_ma_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals