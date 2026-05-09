#!/usr/bin/env python3
# Hypothesis: 12h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above recent bearish fractal high with 1w EMA uptrend and volume > 1.5x average
# Short when price breaks below recent bullish fractal low with 1w EMA downtrend and volume > 1.5x average
# Exit when price crosses the 1w EMA50 in the opposite direction
# Williams Fractals identify potential reversal points; combining with trend and volume filters reduces false signals
# Designed for low-frequency, high-conviction trades on 12h timeframe suitable for trending and ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_WilliamsFractal_Breakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Fractals on 1d: bearish (high) and bullish (low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Bearish fractal: middle high > 2 left and 2 right highs
    bearish = np.zeros(len(df_1d))
    for i in range(2, len(df_1d) - 2):
        if (df_1d['high'].iloc[i] > df_1d['high'].iloc[i-2] and
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i-1] and
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i+1] and
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i+2]):
            bearish[i] = df_1d['high'].iloc[i]
    
    # Bullish fractal: middle low < 2 left and 2 right lows
    bullish = np.zeros(len(df_1d))
    for i in range(2, len(df_1d) - 2):
        if (df_1d['low'].iloc[i] < df_1d['low'].iloc[i-2] and
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i-1] and
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i+1] and
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i+2]):
            bullish[i] = df_1d['low'].iloc[i]
    
    # Align fractals to 12h with 2-bar delay for confirmation (fractals need 2 bars after to confirm)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA and fractal confirmation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above confirmed bearish fractal, EMA50 uptrend, volume spike
            if (close[i] > bearish_aligned[i] and bearish_aligned[i] > 0 and
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below confirmed bullish fractal, EMA50 downtrend, volume spike
            elif (close[i] < bullish_aligned[i] and bullish_aligned[i] > 0 and
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1w EMA50
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1w EMA50
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals