#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above recent bearish fractal (resistance) + price > 1d EMA34 + volume > 1.5x 20-period EMA of volume
# Short when price breaks below recent bullish fractal (support) + price < 1d EMA34 + volume > 1.5x 20-period EMA of volume
# Williams Fractals identify key support/resistance levels; EMA34 filters counter-trend trades; volume confirms breakout strength
# Designed for 12h timeframe to target 15-25 trades/year (60-100 total over 4 years)
# Focus on breakouts from proven swing points reduces whipsaw vs. indicator crossovers

name = "12h_Williams_Fractal_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    # Calculate Williams Fractals on 1d timeframe
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d))
    bullish_fractal = np.zeros(len(low_1d))
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra bars for confirmation (after the center bar forms)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period EMA of 1d volume
        vol_filter = volume[i] > 1.5 * vol_ema_20_aligned[i]
        
        # Fractal breakout conditions
        # Long: price breaks above recent bearish fractal (resistance)
        # Short: price breaks below recent bullish fractal (support)
        resistance_level = bearish_fractal_aligned[i]
        support_level = bullish_fractal_aligned[i]
        
        # Only consider valid fractal levels (non-zero)
        resistance_valid = resistance_level > 0
        support_valid = support_level > 0
        
        if position == 0:
            # Look for entry: fractal breakout + trend + volume
            long_condition = resistance_valid and close[i] > resistance_level and close[i] > ema_34_aligned[i] and vol_filter
            short_condition = support_valid and close[i] < support_level and close[i] < ema_34_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below support level or trend fails
            if support_valid and close[i] < support_level:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_aligned[i]:  # trend failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above resistance level or trend fails
            if resistance_valid and close[i] > resistance_level:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_aligned[i]:  # trend failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals