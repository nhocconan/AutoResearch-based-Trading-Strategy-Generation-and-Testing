#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation
# Uses 1-week EMA200 for primary trend direction (bull/bear regime)
# Enters on break of confirmed Williams fractal (bullish/bearish) in direction of weekly trend
# Volume confirmation > 1.8x average to filter weak breakouts
# Exits on opposite fractal break or trend reversal
# Designed for low frequency (target 12-37 trades/year) with high edge in both bull and bear markets

name = "6h_WilliamsFractal_1wEMA200_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get weekly data for EMA200 trend filter (primary trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Williams Fractals on daily data
    # Bearish fractal: high[n] > high[n-2], high[n] > high[n-1], high[n] > high[n+1], high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2], low[n] < low[n-1], low[n] < low[n+1], low[n] < low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Initialize fractal arrays
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Calculate fractals (need 2 bars on each side)
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align fractals to 6h timeframe with additional delay (fractals need confirmation)
    # Williams fractal needs 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 30-period average volume for confirmation
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 200)  # Volume and weekly EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bearish_fractal = bearish_fractal_aligned[i]
        curr_bullish_fractal = bullish_fractal_aligned[i]
        curr_ema200_1w = ema_200_1w_aligned[i]
        curr_vol_ma = vol_ma_30[i]
        
        # Determine trend from weekly EMA200
        uptrend = curr_close > curr_ema200_1w
        downtrend = curr_close < curr_ema200_1w
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: bearish fractal break or trend reversal to downtrend
            if curr_low < curr_bearish_fractal or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish fractal break or trend reversal to uptrend
            if curr_high > curr_bullish_fractal or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 30-period average
            vol_confirmed = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above bullish fractal, uptrend, volume confirmed
            if curr_high > curr_bullish_fractal and uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below bearish fractal, downtrend, volume confirmed
            elif curr_low < curr_bearish_fractal and downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals