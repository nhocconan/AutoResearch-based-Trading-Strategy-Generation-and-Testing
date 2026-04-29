#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# Uses 1d Williams Fractals (requires 2-bar confirmation delay) for swing high/low identification
# Price breaking above recent 1d bullish fractal with 1d EMA34 uptrend and volume spike = long
# Price breaking below recent 1d bearish fractal with 1d EMA34 downtrend and volume spike = short
# Exits on opposite fractal break or trend reversal
# Designed for ~12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# Williams Fractals provide natural support/resistance with built-in confirmation delay
# Trend filter (1d EMA34) ensures we only trade with higher timeframe momentum
# Volume confirmation (>1.5x average) reduces false breakouts
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34

name = "6h_WilliamsFractal_Breakout_1dEMA34_Trend_Volume"
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
    
    # Get 1d data for Williams Fractals and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for fractals
        return np.zeros(n)
    
    # Calculate 1d Williams Fractals
    # Bearish fractal: high[n] is highest among high[n-2], high[n-1], high[n], high[n+1], high[n+2]
    # Bullish fractal: low[n] is lowest among low[n-2], low[n-1], low[n], low[n+1], low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: current high is highest of surrounding 2 bars each side
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: current low is lowest of surrounding 2 bars each side
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe with proper delay
    # Williams Fractals need 2 extra bars for confirmation (formation + 2 subsequent closes)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)  # EMA only needs 1-bar delay
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume MA and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bearish_fractal = bearish_fractal_aligned[i]
        curr_bullish_fractal = bullish_fractal_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price breaks below bullish fractal (support) OR trend turns down
            if curr_low < curr_bullish_fractal or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above bearish fractal (resistance) OR trend turns up
            if curr_high > curr_bearish_fractal or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 1.5x 20-period average
            vol_spike = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above bearish fractal (resistance) with 1d EMA34 uptrend and volume spike
            if curr_high > curr_bearish_fractal and curr_close > curr_ema34_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below bullish fractal (support) with 1d EMA34 downtrend and volume spike
            elif curr_low < curr_bullish_fractal and curr_close < curr_ema34_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals