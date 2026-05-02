#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout with 1w EMA34 Trend Filter and Volume Confirmation
# Uses weekly Williams Fractals to identify potential reversal points
# Entry logic: Bullish fractal break above weekly high with volume spike in uptrend (price > 1w EMA34) for long
#              Bearish fractal break below weekly low with volume spike in downtrend (price < 1w EMA34) for short
# Works in both bull and bear markets by trading with the 1w trend
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "12h_WilliamsFractal_Breakout_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Williams Fractals
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    
    # Williams Fractals need 2 extra 1w bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Weekly high and low from fractal points
    weekly_high = np.full(len(high_1w), np.nan)
    weekly_low = np.full(len(low_1w), np.nan)
    
    # For bullish fractal (low point), weekly low is the fractal low
    for i in range(len(bullish_fractal)):
        if not np.isnan(bullish_fractal[i]):
            weekly_low[i] = bullish_fractal[i]
    
    # For bearish fractal (high point), weekly high is the fractal high
    for i in range(len(bearish_fractal)):
        if not np.isnan(bearish_fractal[i]):
            weekly_high[i] = bearish_fractal[i]
    
    # Align weekly high/low to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above weekly high AND price > 1w EMA34 (uptrend) AND volume spike
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below weekly low AND price < 1w EMA34 (downtrend) AND volume spike
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1w EMA34 (trend change) OR break below weekly low (reversal)
            if (close[i] < ema_34_1w_aligned[i] or 
                close[i] < weekly_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1w EMA34 (trend change) OR break above weekly high (reversal)
            if (close[i] > ema_34_1w_aligned[i] or 
                close[i] > weekly_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals